import re

def split_into_sentences(text: str) -> list[str]:
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = [s.strip() for s in raw_sentences if s.strip()]
    seen = set()
    unique_sentences = []
    for s in cleaned:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    return unique_sentences