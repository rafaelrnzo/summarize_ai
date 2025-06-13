# import os
# import time
# from typing import List, Dict

# import fitz  # PyMuPDF
# import easyocr

# from langchain_community.document_loaders import (
#     TextLoader, Docx2txtLoader, CSVLoader,
#     UnstructuredExcelLoader, UnstructuredPowerPointLoader,
# )

# from file_type import get_file_type
# from transcribe import transcribe_audio, transcribe_video

# # === File Extension Maps ===

# DOC_LOADERS = {
#     '.docx': Docx2txtLoader,
#     '.doc': Docx2txtLoader,
#     '.csv': CSVLoader,
#     '.xls': UnstructuredExcelLoader,
#     '.xlsx': UnstructuredExcelLoader,
#     '.ppt': UnstructuredPowerPointLoader,
#     '.pptx': UnstructuredPowerPointLoader,
#     '.txt': TextLoader,
#     '.log': TextLoader,
#     '.json': TextLoader,
#     '.xml': TextLoader,
# }

# AUDIO_EXTS = {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}
# VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
# IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

# reader = easyocr.Reader(['en', 'id'], gpu=True)

# # === OCR Functions ===

# def ocr_image(path: str) -> str:
#     results = reader.readtext(path)
#     return "\n".join([res[1] for res in results])


# def extract_text_and_ocr_from_pdf(pdf_path: str) -> str:
#     doc = fitz.open(pdf_path)
#     final_text = []

#     for page_index in range(len(doc)):
#         page = doc.load_page(page_index)
#         page_parts = [f"--- Page {page_index + 1} ---"]

#         # Extract text
#         text = page.get_text().strip()
#         if text:
#             page_parts.append("[Extracted Text]")
#             page_parts.append(text)

#         # OCR on embedded images
#         image_list = page.get_images(full=True)
#         if image_list:
#             page_parts.append("[OCR from Embedded Images]")
#             for img_index, img in enumerate(image_list):
#                 xref = img[0]
#                 try:
#                     base_image = doc.extract_image(xref)
#                     image_bytes = base_image["image"]
#                     results = reader.readtext(image_bytes)
#                     ocr_text = "\n".join(res[1] for res in results)
#                     if ocr_text.strip():
#                         page_parts.append(f"Image {img_index + 1}:\n{ocr_text}")
#                 except Exception as e:
#                     page_parts.append(f"Image {img_index + 1} OCR Failed: {e}")

#         final_text.append("\n".join(page_parts))

#     return "\n\n".join(final_text)

# # === Load and Extract Function ===

# def load_document(file_path: str) -> str:
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File not found: {file_path}")

#     ext, _ = get_file_type(file_path)
#     ext = ext.lower()

#     try:
#         if ext in VIDEO_EXTS:
#             return transcribe_video(file_path)

#         if ext in AUDIO_EXTS:
#             return transcribe_audio(file_path)

#         if ext in IMAGE_EXTS:
#             return ocr_image(file_path)

#         if ext == '.pdf':
#             text = extract_text_and_ocr_from_pdf(file_path)
#             if len(text.strip()) < 50:
#                 print("⚠️ Text extraction minimal, forcing full OCR...")
#                 text = extract_text_and_ocr_from_pdf(file_path)
#             return text

#         if ext in DOC_LOADERS:
#             docs = DOC_LOADERS[ext](file_path).load()
#             text = "\n".join(doc.page_content for doc in docs)
#             if len(text.strip()) < 50:
#                 print("⚠️ Low content extracted, consider checking document formatting.")
#             return text

#         # Fallback generic read
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return f.read()

#     except Exception as e:
#         raise RuntimeError(f"Failed to load {file_path}: {e}")

# # === Batch Process PDFs ===

# def batch_process_pdfs(paths: List[str]) -> Dict[str, str]:
#     results = {}
#     for path in paths:
#         try:
#             results[path] = load_document(path)
#         except Exception as e:
#             results[path] = None
#             print(f"❌ Failed to process {path}: {e}")
#     return results

# # === PDF Info Summary ===

# def get_pdf_info(pdf_path: str) -> Dict[str, any]:
#     info = {
#         "file_path": pdf_path,
#         "file_size": os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0,
#         "text_extractable": False,
#         "ocr_available": True,
#         "recommended_method": "unknown",
#     }

#     try:
#         with fitz.open(pdf_path) as doc:
#             text = "\n".join(page.get_text() for page in doc).strip()
#             info["text_extractable"] = len(text) > 50
#     except Exception:
#         pass

#     info["recommended_method"] = "text_extraction" if info["text_extractable"] else "ocr"
#     return info

# # === Save Output ===

# def save_output(text: str, output_path: str):
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(text)
#     print(f"✅ Output saved to: {output_path}")

# # === Run Test ===

# if __name__ == "__main__":
#     test_path = r"D:\pyproject\proj\summarize_ai\app\vid\jok.mp4"  # ← Ganti path sesuai kebutuhan
#     output_file = "output.txt"  # Simpan hasil ke file output

#     print(f"📄 Processing file: {test_path}")
#     start = time.time()

#     try:
#         raw_text = load_document(test_path)

#         if not raw_text.strip():
#             print("⚠️ Output is empty. File may need OCR fallback or deeper parsing.")
#         else:
#             save_output(raw_text, output_file)

#         elapsed = time.time() - start
#         print(f"\n⏱️ Processing time: {elapsed:.2f} seconds")
#     except Exception as e:
#         print(f"❌ Error: {e}")
