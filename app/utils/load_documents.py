import os

from langchain_community.document_loaders import (
    TextLoader, Docx2txtLoader, CSVLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader
)

from utils.extract_ocr import extract_text_ocr, ocr_image
from utils.image_reader import describe_image, describe_image_and_save
from utils.file_type import get_file_type
from utils.transcribe import transcribe_audio, transcribe_video

OUTPUT_FOLDER = "assets/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DOC_LOADERS = {
    '.docx': Docx2txtLoader,
    '.doc': Docx2txtLoader,
    '.csv': CSVLoader,
    '.xls': UnstructuredExcelLoader,
    '.xlsx': UnstructuredExcelLoader,
    '.ppt': UnstructuredPowerPointLoader,
    '.pptx': UnstructuredPowerPointLoader,
    '.txt': TextLoader,
    '.log': TextLoader,
    '.json': TextLoader,
    '.xml': TextLoader,
}

AUDIO_EXTS = {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}


def load_document(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext, _ = get_file_type(file_path)
    ext = ext.lower()

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(OUTPUT_FOLDER, f"{file_name}_output.txt")

    try:
        if ext in VIDEO_EXTS:
            content = transcribe_video(file_path)
        elif ext in AUDIO_EXTS:
            content = transcribe_audio(file_path)
        elif ext == '.png':
            output_file = describe_image_and_save(file_path)
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()

        elif ext in IMAGE_EXTS:
            content = ocr_image(file_path)
        elif ext == '.pdf':
            content = extract_text_ocr(file_path)
        elif ext in DOC_LOADERS:
            docs = DOC_LOADERS[ext](file_path).load()
            content = "\n".join(doc.page_content for doc in docs)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(content)

        return content

    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}")
