import base64
import json
import os
from io import BytesIO
from pathlib import Path
from PIL import Image
import requests

MODEL_URL = os.getenv("MODEL_API_MULTI")
MODEL_QWEN_MULTI = os.getenv("MODEL_QWEN_MULTI", "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit")
OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", "assets/"))
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def describe_image(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        payload = {
            "model": MODEL_QWEN_MULTI,
            "messages": [
                {
                    "role": "system",
                    "content": "Kamu adalah asisten AI yang mendeskripsikan gambar secara rinci dalam bahasa Indonesia."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Deskripsikan gambar ini secara sangat detail dan lengkap dalam Bahasa Indonesia."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 1.0
        }

        response = requests.post(
            f"{MODEL_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        description = result["choices"][0]["message"]["content"].strip()
        return description

    except Exception as e:
        raise RuntimeError(f"Gagal mendeskripsikan gambar dengan LLM: {e}")

def describe_image_and_save(image_path: str) -> str:

    description = describe_image(image_path)

    file_stem = Path(image_path).stem
    output_path = OUTPUT_FOLDER / f"{file_stem}_output.txt"

    with output_path.open("w", encoding="utf-8") as f:
        f.write(description)

    return output_path.as_posix()

describe_image_with_llm = describe_image
