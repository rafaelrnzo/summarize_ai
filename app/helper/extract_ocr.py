import requests
import os

EXTRACT_API_URL = os.getenv("SUMMARIZE_SERVICE_URL")

def extract_text_ocr(pdf_path: str) -> str:
    url = f"{EXTRACT_API_URL}/extract/pdf"
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            return response.json().get("text", "")
        except Exception as e:
            raise RuntimeError(f"OCR microservice failed: {e}")
