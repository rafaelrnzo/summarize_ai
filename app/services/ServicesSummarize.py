import re
import traceback
import logging
import asyncio
import aiohttp
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import List, Dict, Any
from utils.clutering_text import (
    truncate_text_to_tokens, preprocess_text, 
    split_into_sentences, create_sentence_clusters
)
from utils.load_documents import load_document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
MAX_TOKENS_PROMPT = int(os.getenv("MAX_TOKENS_PROMPT"))
OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", "assets/"))
VLLM_ENDPOINT = os.getenv("MODEL_API_BASE")
MODEL_MISTRAL = os.getenv("MODEL_MISTRAL")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

class VLLMClient:
    def __init__(self, endpoint: str, model: str, max_concurrent: int = 10):
        self.endpoint = endpoint
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=120, connect=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate_text(self, prompt: str, max_tokens: int = 512) -> str:
        async with self.semaphore:
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": max_tokens,
                    "stream": False
                }
                async with self.session.post(f"{self.endpoint}/chat/completions", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    error_text = await response.text()
                    logger.error(f"VLLM API error {response.status}: {error_text}")
                    raise Exception(f"VLLM API error: {response.status}")
            except asyncio.TimeoutError:
                logger.error("Request timed out")
                raise Exception("Request timeout")
            except Exception as e:
                logger.error(f"Error in generate_text: {e}")
                raise


def clean_output_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'\n+', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def create_summary_prompt(text: str, max_words: int = 512) -> str:
    return f"""<|im_start|>system
You're helping someone understand this content by creating a natural, conversational summary. Write like you're explaining it to a friend - use everyday language, avoid jargon, and make it engaging. Keep it under {max_words} words and focus on what's actually important and interesting.
<|im_end|>

<|im_start|>user
Can you help me understand what this text is about? Here it is:

{text}
<|im_end|>

<|im_start|>assistant
Here's what this is about:"""

def create_cluster_summary_prompt(text: str) -> str:
    return f"""<|im_start|>system
Explain the main points from this text in a natural, conversational way. Keep it brief (under 100 words) and focus on what's most interesting or important. Write like you're telling someone about it over coffee.
<|im_end|>

<|im_start|>user
What are the key points here?

{text}
<|im_end|>

<|im_start|>assistant
The main points are:"""

def create_final_summary_prompt(summaries: str) -> str:
    return f"""<|im_start|>system
You've got several key points from different sections. Now bring them together into one clear, engaging summary that tells the complete story. Write naturally - like you're giving someone the highlights of something you just read. Keep it under 512 words and make sure it flows well from start to finish.
<|im_end|>

<|im_start|>user
Here are the main points from different sections. Can you put this all together for me?

{summaries}
<|im_end|>

<|im_start|>assistant
Here's the complete picture:"""

async def summarize_text_async(text: str, vllm_client: VLLMClient) -> str:
    if not text:
        return "No text available to summarize."
    prompt = create_summary_prompt(truncate_text_to_tokens(text, MAX_TOKENS_PROMPT))
    try:
        response = await vllm_client.generate_text(prompt, max_tokens=400)
        return clean_output_text(response)
    except Exception as e:
        logger.error(f"Error in summarize_text_async: {e}")
        return clean_output_text(f"[Auto-extracted summary] {text[:300]}...")

async def get_cluster_summaries_async(clusters: Dict[Any, List[str]], vllm_client: VLLMClient) -> List[str]:
    async def summarize_cluster(cluster_id, sentences):
        text = " ".join(sentences)
        prompt = create_cluster_summary_prompt(truncate_text_to_tokens(text, MAX_TOKENS_PROMPT))
        try:
            summary = await vllm_client.generate_text(prompt, max_tokens=200)
            logger.info(f"Successfully summarized cluster {cluster_id}")
            return clean_output_text(summary)
        except Exception as e:
            logger.error(f"Error summarizing cluster {cluster_id}: {e}")
            fallback = " ".join([sentences[0], sentences[len(sentences)//2], sentences[-1]]) if len(sentences) > 3 else " ".join(sentences[:1] + sentences[-1:])
            return clean_output_text(f"[Auto-extracted summary] {fallback[:300]}...")

    tasks = [summarize_cluster(cluster_id, sentences) for cluster_id, sentences in clusters.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [s if not isinstance(s, Exception) else "[Summary generation failed]" for s in results]

async def create_final_summary_async(cluster_summaries: List[str], vllm_client: VLLMClient) -> str:
    if not cluster_summaries:
        return "No summary available due to processing errors."

    if len(cluster_summaries) > 5:
        chunk_size = max(1, len(cluster_summaries) // 3)
        cluster_summaries = [
            "Section summary: " + " ".join(cluster_summaries[i:i+chunk_size])
            for i in range(0, len(cluster_summaries), chunk_size)
        ]

    prompt = create_final_summary_prompt(truncate_text_to_tokens(
        " ".join([f"Section {i+1}: {s}" for i, s in enumerate(cluster_summaries)]),
        MAX_TOKENS_PROMPT
    ))

    try:
        response = await vllm_client.generate_text(prompt, max_tokens=400)
        return clean_output_text(response)
    except Exception as e:
        logger.error(f"Error generating final summary: {e}")
        return clean_output_text("SUMMARY COMPILATION: " + " ".join(cluster_summaries[:3]))

async def summarize_with_clustering_async(text: str, num_clusters: int = 5) -> str:
    if not text:
        return "No text available to summarize."
    sentences = split_into_sentences(preprocess_text(text))
    logger.info(f"{len(sentences)} sentences found.")

    async with VLLMClient(VLLM_ENDPOINT, MODEL_MISTRAL, MAX_CONCURRENT_REQUESTS) as vllm_client:
        if len(sentences) < 10:
            return await summarize_text_async(" ".join(sentences), vllm_client)

        clusters = create_sentence_clusters(sentences, num_clusters)
        logger.info(f"Created {len(clusters)} clusters")

        summaries = await get_cluster_summaries_async(clusters, vllm_client)
        return await create_final_summary_async(summaries, vllm_client)

async def process_document(file_path: str) -> Dict[str, Any]:
    try:
        logger.info(f"Loading document: {file_path}")
        load_document(file_path)
        file_path_obj = Path(file_path)
        output_path = OUTPUT_FOLDER / f"{file_path_obj.stem}_output.txt"

        with output_path.open('r', encoding='utf-8') as f:
            document_text = f.read()

        word_count = len(document_text.split())
        logger.info(f"Document contains approximately {word_count} words.")
        num_clusters = max(3, min(15, word_count // 600))
        logger.info(f"Using {num_clusters} clusters for summarization.")

        summary = await summarize_with_clustering_async(document_text, num_clusters)

        result = {
            "text": document_text,
            "word_count": word_count,
            "summary": clean_output_text(summary),
            "language": "en",
            "processing_info": {
                "clusters_used": num_clusters,
                "concurrent_requests": MAX_CONCURRENT_REQUESTS,
                "vllm_model": MODEL_MISTRAL
            }
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

async def process_multiple_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    tasks = [process_document(file_path) for file_path in file_paths]
    return await asyncio.gather(*tasks, return_exceptions=True)
