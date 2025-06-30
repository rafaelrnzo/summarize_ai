import re
import traceback
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
from utils.clutering_text import (
    truncate_text_to_tokens, preprocess_text, 
    split_into_sentences, create_sentence_clusters
)
from utils.load_documents import load_document
from core.depedencies import create_llm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
MAX_TOKENS_PROMPT = int(os.getenv("MAX_TOKENS_PROMPT"))
OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", "assets/"))  

def clean_output_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'\n+', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def summarize_text(text: str, llm) -> str:
    if not text:
        return "No text available to summarize."
    truncated_text = truncate_text_to_tokens(text, MAX_TOKENS_PROMPT)
    messages = [
        SystemMessage(content="Summarize the following text into a concise, well-structured paragraph. Provide a clean output without special formatting or line breaks:"),
        HumanMessage(content=truncated_text)
    ]
    try:
        response = llm.invoke(messages).content
        return clean_output_text(response)
    except Exception as e:
        logger.error(f"Error in summarize_text: {e}")
        return clean_output_text(f"[Auto-extracted summary] {text[:300]}...")

def get_cluster_summaries(clusters: dict, llm) -> list[str]:
    summaries = []
    for cluster_id, sentences in clusters.items():
        text = " ".join(sentences)
        truncated = truncate_text_to_tokens(text, MAX_TOKENS_PROMPT)
        messages = [
            SystemMessage(content="Summarize this text into a short, concise paragraph without line breaks:"),
            HumanMessage(content=truncated)
        ]
        try:
            summary = llm.invoke(messages).content
            logger.info(f"Successfully summarized cluster {cluster_id}")
            summaries.append(clean_output_text(summary))
        except Exception as e:
            logger.error(f"Error summarizing cluster {cluster_id}: {e}")
            fallback = " ".join(sentences[:1] + sentences[-1:]) if len(sentences) <= 3 else " ".join([sentences[0], sentences[len(sentences)//2], sentences[-1]])
            summaries.append(clean_output_text(f"[Auto-extracted summary] {fallback[:300]}..."))
    return summaries

def create_final_summary(cluster_summaries: list[str], llm) -> str:
    if not cluster_summaries:
        return "Tidak ada ringkasan yang tersedia karena terjadi kesalahan saat pemrosesan."

    if len(cluster_summaries) > 5:
        chunk_size = max(1, len(cluster_summaries) // 3)
        cluster_summaries = [
            "Ringkasan bagian: " + " ".join(cluster_summaries[i:i+chunk_size])
            for i in range(0, len(cluster_summaries), chunk_size)
        ]

    full_text = " ".join([f"Bagian {i+1}: {s}" for i, s in enumerate(cluster_summaries)])
    truncated = truncate_text_to_tokens(full_text, MAX_TOKENS_PROMPT)
    messages = [
        SystemMessage(content="Gabungkan dan rangkum poin-poin penting dari teks berikut menjadi satu kesimpulan yang koheren dalam paragraf yang terhubung. Fokus pada informasi inti, jaga alur logis, dan hindari pengulangan. Gunakan bahasa Indonesia yang jelas dan ringkas. Jangan gunakan line breaks atau formatting khusus."),
        HumanMessage(content=truncated)
    ]
    try:
        return clean_output_text(llm.invoke(messages).content)
    except Exception as e:
        logger.error(f"Error generating final summary: {e}")
        return clean_output_text("KOMPILASI RINGKASAN: " + " ".join(cluster_summaries[:3]))

def summarize_with_clustering(text: str, num_clusters: int = 5) -> str:
    if not text:
        return "No text available to summarize."
    logger.info("Cleaning and splitting text...")
    sentences = split_into_sentences(preprocess_text(text))
    logger.info(f"{len(sentences)} sentences found.")
    llm = create_llm()

    if len(sentences) < 10:
        return summarize_text(" ".join(sentences), llm)

    clusters = create_sentence_clusters(sentences, num_clusters)
    summaries = get_cluster_summaries(clusters, llm)
    return create_final_summary(summaries, llm)

def process_document(file_path: str):
    try:
        logger.info(f"Loading document: {file_path}")
        load_document(file_path)

        file_path_obj = Path(file_path)
        output_path = OUTPUT_FOLDER / f"{file_path_obj.stem}_output.txt"

        with output_path.open('r', encoding='utf-8') as f:
            document_text = f.read()

        word_count = len(document_text.split())
        logger.info(f"Document contains approximately {word_count} words.")
        num_clusters = max(3, min(10, word_count // 800))
        logger.info(f"Using {num_clusters} clusters for summarization.")

        final_summary = summarize_with_clustering(document_text, num_clusters)

        result = {
            "text": document_text,
            "word_count": word_count,
            "summary": clean_output_text(final_summary),
        }

        if file_path_obj.suffix.lower() in {".mp3", ".mp4"}:
            with output_path.open('r', encoding='utf-8') as f:
                lines = [clean_output_text(line) for line in f.readlines() if line.strip()]
            result["scripts"] = [{"text": line} for line in lines]

        logger.info("Document processed successfully")

        try:
            output_path.unlink()
            logger.info(f"Deleted temporary output file: {output_path}")
        except Exception as del_err:
            logger.warning(f"Failed to delete output file: {del_err}")

        return result

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        logger.debug(traceback.format_exc())
        return {"error": f"Error processing document: {e}"}
