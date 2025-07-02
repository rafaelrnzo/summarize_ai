import os
import requests
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

TRANSCRIBE_API_URL = os.getenv("WHISPER_API_URL", "http://192.168.100.3:8001")

def transcribe_audio(audio_path: str) -> str:
    url = f"{TRANSCRIBE_API_URL}/transcribe/audio"
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "application/octet-stream")}
            response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription error: {str(e)}")

def transcribe_video(video_path: str) -> str:
    url = f"{TRANSCRIBE_API_URL}/transcribe/video"
    try:
        with open(video_path, "rb") as f:
            files = {"file": (os.path.basename(video_path), f, "application/octet-stream")}
            response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video transcription error: {str(e)}")
