import whisper
from dotenv import load_dotenv
from fastapi import HTTPException
import os

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

def transcribe_audio(audio_path: str) -> str:
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription error: {str(e)}")

def transcribe_video(video_path: str) -> str:
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(video_path)
        return result['text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video transcription error: {str(e)}")
