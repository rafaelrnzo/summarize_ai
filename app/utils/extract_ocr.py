import fitz  # PyMuPDF
import easyocr

reader = easyocr.Reader(['en'], gpu=True)

def ocr_image(path: str) -> str:
    results = reader.readtext(path)
    return "\n".join([res[1] for res in results])

def extract_text_ocr(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    final_text = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        page_parts = [f"--- Page {page_index + 1} ---"]

        text = page.get_text().strip()
        if text:
            page_parts.append("[Extracted Text]")
            page_parts.append(text)

        image_list = page.get_images(full=True)
        if image_list:
            page_parts.append("[OCR from Embedded Images]")
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    results = reader.readtext(image_bytes)
                    ocr_text = "\n".join(res[1] for res in results)
                    if ocr_text.strip():
                        page_parts.append(f"Image {img_index + 1}:\n{ocr_text}")
                except Exception as e:
                    page_parts.append(f"Image {img_index + 1} OCR Failed: {e}")

        final_text.append("\n".join(page_parts))

    return "\n\n".join(final_text)