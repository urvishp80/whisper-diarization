import os
from helpers import *
from whisper import load_model
import whisperx
import torch
import librosa
import soundfile
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import time


def diarize_audio(audio_file_name, whisper_model_name, stemming):
    try:
        start_time = time.time()

        if stemming:
            # Isolate vocals from the rest of the audio

            return_code = os.system(
                f'python3 -m demucs.separate -n htdemucs_ft --two-stems=vocals "{audio_file_name}" -o "temp_outputs"'
            )

            if return_code != 0:
                print(
                    "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
                )
                vocal_target = audio_file_name
            else:
                vocal_target = f"temp_outputs/htdemucs_ft/{os.path.basename(audio_file_name)[:-4]}/vocals.wav"
        else:
            vocal_target = audio_file_name


        # Large models result in considerably better and more aligned (words, timestamps) mapping.
        whisper_model = load_model(whisper_model_name)
        whisper_results = whisper_model.transcribe(vocal_target, beam_size=None, verbose=False)

        # clear gpu vram
        del whisper_model
        torch.cuda.empty_cache()

        device = "cuda"
        alignment_model, metadata = whisperx.load_align_model(
            language_code=whisper_results["language"], device=device
        )
        result_aligned = whisperx.align(
            whisper_results["segments"], alignment_model, metadata, vocal_target, device
        )

        # clear gpu vram
        del alignment_model
        torch.cuda.empty_cache()

        # convert audio to mono for NeMo combatibility
        signal, sample_rate = librosa.load(vocal_target, sr=None)
        ROOT = os.getcwd()
        temp_path = os.path.join(ROOT, "temp_outputs")
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
        os.chdir(temp_path)
        soundfile.write("mono_file.wav", signal, sample_rate, "PCM_24")

        # Initialize NeMo MSDD diarization model
        msdd_model = NeuralDiarizer(cfg=create_config())
        msdd_model.diarize()

        del msdd_model
        torch.cuda.empty_cache()

        # Reading timestamps <> Speaker Labels mapping

        output_dir = "nemo_outputs"

        speaker_ts = []
        with open(f"{output_dir}/pred_rttms/mono_file.rttm", "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        wsm = get_words_speaker_mapping(result_aligned["word_segments"], speaker_ts, "start")

        if whisper_results["language"] in punct_model_langs:
            # restoring punctuation in the transcript to help realign the sentences
            punct_model = PunctuationModel(model="kredor/punctuate-all")

            words_list = list(map(lambda x: x["word"], wsm))

            labled_words = punct_model.predict(words_list)

            ending_puncts = ".?!"
            model_puncts = ".,;:!?"

            # We don't want to punctuate U.S.A. with a period. Right?
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word

            wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        else:
            print(
                f'Punctuation restoration is not available for {whisper_results["language"]} language.'
            )

        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        os.chdir(ROOT)  # back to parent dir

        output_dir = "output_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir,f"{os.path.basename(audio_file_name)[:-4]}.srt")
        with open(output_path, "w", encoding="utf-8-sig") as srt:
            write_srt(ssm, srt)

        # Upload the srt file to s3
        uploaded = upload_file_to_s3(file_name=output_path, bucket=bucket, folder_name=srt_folder_name)
        if uploaded:
            print(f"{os.path.basename(output_path)} uploaded successfully to s3")
        else:
            print(f"{os.path.basename(output_path)} not uploaded to s3 due to some error")

        cleanup(temp_path)
        cleanup(output_dir)
        total_time = time.time() - start_time
        print(f"Total time taken {total_time}")

    except Exception as e:
        print(str(e))
