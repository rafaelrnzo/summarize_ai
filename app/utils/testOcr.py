import fitz  # PyMuPDF
import easyocr

def extract_text_from_pdf(pdf_path: str, lang: str = 'en') -> str:
    reader = easyocr.Reader([lang])
    doc = fitz.open(pdf_path)
    extracted_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")

        results = reader.readtext(img_bytes)
        page_text = "\n".join([res[1] for res in results])
        extracted_text.append(f"--- Page {page_num + 1} ---\n{page_text}")

    return "\n\n".join(extracted_text)

if __name__ == "__main__":
    pdf_path = r"D:\pyproject\proj\summarize_ai\app\vid\testOCR.pdf"  # Windows path
    raw_text = extract_text_from_pdf(pdf_path)
    print(raw_text)
