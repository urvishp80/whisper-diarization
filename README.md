## This Repo is cloned from:- https://github.com/MahmoudAshraf97/whisper-diarization

### To run the whisper diarization on all audio files from s3

It will download audio file from s3, process it and upload the srt file to s3.

Required python>=3.9 and install all dependencies using:
- pip install requirements.txt
- pip install transformers==4.26.1

we have to install transformer separately due to versioning error with nemo lib

Set up environment variables: Create .env file in the root folder and add following keys -
```commandline
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
BUCKET_NAME = ""
```

## Usage:-
if you want stemming=True and model_name="medium.en" as deafult use:-
```commandline
python main.py
```
else give the args for example:-
```commandline
python main.py --no-stem --whisper-model "large.en"
```