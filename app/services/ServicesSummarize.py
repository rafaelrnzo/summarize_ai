import os 
import re 
from langchain_core.messages import HumanMessage, SystemMessage 
from utils.clutering_text import truncate_text_to_tokens, preprocess_text, split_into_sentences, create_sentence_clusters
from utils.load_documents import load_document
from core.depedencies import create_llm
import traceback
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
MAX_TOKENS_PROMPT = int(os.getenv("MAX_TOKENS_PROMPT"))
OUTPUT_TEXT = os.getenv("OUTPUT_PATH")

def clean_output_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'\n+', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

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
        fallback = text[:300]
        return clean_output_text(f"[Auto-extracted summary] {fallback}...")

def get_cluster_summaries(clusters: dict, llm) -> list[str]:
    summaries = []
    
    for cluster_id, sentences in clusters.items():
        joined_text = " ".join(sentences)
        truncated_joined_text = truncate_text_to_tokens(joined_text, MAX_TOKENS_PROMPT)
        
        messages = [
            SystemMessage(content="Summarize this text into a short, concise paragraph without line breaks:"),
            HumanMessage(content=truncated_joined_text)
        ]

        try:
            summary = llm.invoke(messages).content
            logger.info(f"Successfully summarized cluster {cluster_id}")
            summaries.append(clean_output_text(summary))
        except Exception as e:
            logger.error(f"Error summarizing cluster {cluster_id}: {e}")
            logger.info("Falling back to extractive summarization...")
            
            all_sents = sentences
            if len(all_sents) > 3:
                key_sents = [
                    all_sents[0],  
                    all_sents[len(all_sents)//2], 
                    all_sents[-1]
                ]
                fallback = " ".join(key_sents)
            else:
                fallback = " ".join(all_sents[:2])  
            
            summaries.append(clean_output_text(f"[Auto-extracted summary] {fallback[:300]}..."))
    
    return summaries

def create_final_summary(cluster_summaries: list[str], llm) -> str:
    if not cluster_summaries:
        return "Tidak ada ringkasan yang tersedia karena terjadi kesalahan saat pemrosesan."
    
    if len(cluster_summaries) > 5:
        logger.info(f"Terlalu banyak ringkasan cluster ({len(cluster_summaries)}). Menggabungkan...")
        chunk_size = max(1, len(cluster_summaries) // 3)
        merged_summaries = []
        
        for i in range(0, len(cluster_summaries), chunk_size):
            chunk = cluster_summaries[i:i+chunk_size]
            merged = "Ringkasan bagian: " + " ".join(chunk)
            merged_summaries.append(merged)
        
        cluster_summaries = merged_summaries
    
    all_summaries = " ".join([f"Bagian {i+1}: {s}" for i, s in enumerate(cluster_summaries)])
    truncated_summaries = truncate_text_to_tokens(all_summaries, MAX_TOKENS_PROMPT)

    messages = [
        SystemMessage(content="Gabungkan dan rangkum poin-poin penting dari teks berikut menjadi satu kesimpulan yang koheren dalam paragraf yang terhubung. Fokus pada informasi inti, jaga alur logis, dan hindari pengulangan. Gunakan bahasa Indonesia yang jelas dan ringkas. Jangan gunakan line breaks atau formatting khusus."),
        HumanMessage(content=truncated_summaries)
    ]

    try:
        final_summary = llm.invoke(messages).content
        return clean_output_text(final_summary)
    except Exception as e:
        logger.error(f"Error generating final summary: {e}")
        fallback = " ".join(cluster_summaries[:3])
        return clean_output_text("KOMPILASI RINGKASAN: " + fallback)

def summarize_with_clustering(text: str, num_clusters: int = 5) -> str:
    if not text:
        return "No text available to summarize."
        
    logger.info("Cleaning text...")
    cleaned_text = preprocess_text(text)
    
    logger.info("Splitting into sentences...")
    sentences = split_into_sentences(cleaned_text)
    logger.info(f"{len(sentences)} sentences found.")

    llm = create_llm()

    if len(sentences) < 10:
        logger.info("Text is short, summarizing directly.")
        return summarize_text(cleaned_text, llm)

    logger.info("Clustering sentences...")
    clusters = create_sentence_clusters(sentences, num_clusters)
    logger.info(f"Created {len(clusters)} clusters/chunks.")

    logger.info("Summarizing each cluster...")
    summaries = get_cluster_summaries(clusters, llm)

    logger.info("Creating final summary...")
    final_summary = create_final_summary(summaries, llm)
    return final_summary

def process_document(file_path):
    try:
        logger.info(f"Loading document: {file_path}")
        load_document(file_path)  

        with open(OUTPUT_TEXT, 'r', encoding='utf-8') as f:
            document_text = f.read()

        word_count = len(document_text.split())
        logger.info(f"Document contains approximately {word_count} words.")

        num_clusters = max(3, min(10, word_count // 800))
        logger.info(f"Using {num_clusters} clusters for summarization.")

        final_summary = summarize_with_clustering(document_text, num_clusters)

        clean_summary = clean_output_text(final_summary)

        return {
            "text": document_text,
            "word_count": word_count,
            "summary": clean_summary
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing document: {error_message}")
        logger.debug(traceback.format_exc())
        return {"error": f"Error processing document: {error_message}"}