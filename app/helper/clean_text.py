import re 


def clean_output_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'\n+', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()