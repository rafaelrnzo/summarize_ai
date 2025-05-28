import whisper
from dotenv import load_dotenv
import os

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny") 

def transcribe_audio(audio_path):
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"Audio transcription error: {str(e)}")
        return f"[Error transcribing audio: {str(e)}]"

def transcribe_video(video_path):
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(video_path)
        return result['text']
    except Exception as e:
        print(f"Video transcription error: {str(e)}")
        return f"[Error transcribing video: {str(e)}]"
