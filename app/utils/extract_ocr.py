from http.client import HTTPException
import fitz  # PyMuPDF
import easyocr
import os 


reader = easyocr.Reader(['en'], gpu=True)

EXTRACT_API_URL = os.getenv("SUMMARIZE_SERVICE_URL")

def ocr_image(path: str) -> str:
    results = reader.readtext(path)
    return "\n".join([res[1] for res in results])

def extract_text_ocr(pdf_path: str) -> str:
    url = f"{EXTRACT_API_URL}/extract/pdf"
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")