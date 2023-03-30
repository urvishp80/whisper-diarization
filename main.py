from diarize import *
from helpers import *
import argparse


def start_diarize(whisper_model_name, stemming):
    input_data = "input_data"
    os.makedirs(input_data, exist_ok=True)
    audio_file_list = get_all_files_from_s3(bucket, input_folder_name)
    srt_file_list = get_all_files_from_s3(bucket, srt_folder_name)
    len_audio_file_list = len(audio_file_list)
    len_srt_file_list = len(srt_file_list)

    while len_audio_file_list != len_srt_file_list:
        for audio_file in audio_file_list:
            print(audio_file)
            try:
                if srt_folder_name+"/"+str(os.path.basename(audio_file)[:-4] + ".srt") not in srt_file_list:
                    audio_file_name = download_file_from_s3(audio_file, bucket, input_folder_name)
                    print(f"Input file name:- {audio_file_name}")
                    if audio_file_name:
                        diarize_audio(audio_file_name, whisper_model_name, stemming)
                    else:
                        print(f"There is problem in downloading file from the s3")
                else:
                    print(f"{audio_file} is already processed and output is saved in s3")
            except Exception as e:
                print(str(e))
                print(f"error ocurred in {audio_file}")

        srt_file_list = get_all_files_from_s3(bucket, srt_folder_name)
        len_srt_file_list = len(srt_file_list)
        # Checking during processing any new file uploaded if uploaded it will update len and list
        audio_file_list = get_all_files_from_s3(bucket, input_folder_name)
        len_audio_file_list = len(audio_file_list)

    print(f"Diarization completed and total {len_srt_file_list} srt files inserted")
    cleanup(input_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help="Disables source separation."
             "This helps with long files that don't contain a lot of music.",
    )
    parser.add_argument(
        "--whisper-model",
        dest="model_name",
        default="medium.en",
        help="name of the Whisper model to use",
    )
    args = parser.parse_args()
    start_diarize(args.model_name, args.stemming)