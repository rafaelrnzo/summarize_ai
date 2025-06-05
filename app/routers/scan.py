from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import FileResponse
from typing import List
from PIL import Image
import os
import uuid
import subprocess

router = APIRouter(prefix="/v1/scan", tags=["scanner"])

@router.post("/img-to-pdf", response_class=FileResponse)
async def convert_images_to_pdf(
    files: List[UploadFile] = File(...),
    output_name: str = Query(..., description="Output PDF file name without .pdf extension")
):
    session_id = str(uuid.uuid4())
    temp_dir = f"temp/{session_id}"
    os.makedirs(temp_dir, exist_ok=True)

    image_paths = []

    for i, file in enumerate(files):
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            return {"error": f"Unsupported file format: {file.filename}"}

        path = os.path.join(temp_dir, f"img_{i}{ext}")
        with open(path, "wb") as f:
            f.write(await file.read())
        image_paths.append(path)

    temp_pdf_path = os.path.join(temp_dir, "temp_combined.pdf")
    images = [Image.open(p).convert("RGB") for p in image_paths]
    images[0].save(temp_pdf_path, save_all=True, append_images=images[1:])

    output_pdf_path = os.path.join(
        r"D:\pyproject\proj\summarize_ai\app\file",
        f"{output_name}.pdf"
    )

    subprocess.run([
        "ocrmypdf", temp_pdf_path, output_pdf_path, "--language", "eng"
    ], check=True)

    return FileResponse(
        output_pdf_path,
        media_type="application/pdf",
        filename=f"{output_name}.pdf"
    )
