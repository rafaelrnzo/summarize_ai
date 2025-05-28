import os
import whisper
import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    YoutubeLoader,
)

import mimetypes
from pathlib import Path
import pandas as pd
import traceback

# Constants
MODEL_API_BASE = "http://192.168.100.3:8000/v1"
WHISPER_MODEL = "tiny"  
MAX_TOKENS_PER_CLUSTER = 500  
MAX_TOKENS_PROMPT = 800  
MAX_TOKENS_MODEL = 1600  

def ensure_nltk_resources():
    for resource in ['punkt']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

ensure_nltk_resources()

def get_file_type(file_path):
    file_extension = Path(file_path).suffix.lower()
    mime_type, _ = mimetypes.guess_type(file_path)
    
    return file_extension, mime_type

def load_document(file_path):
    file_extension, mime_type = get_file_type(file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Processing file: {file_path}")
    print(f"Detected extension: {file_extension}, MIME type: {mime_type}")
    
    try:
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return transcribe_video(file_path)
            
        elif file_extension in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
            return transcribe_audio(file_path)
            
        elif file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
            
        elif file_extension in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
            
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
            
        elif file_extension in ['.xls', '.xlsx']:
            loader = UnstructuredExcelLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
            
        elif file_extension in ['.ppt', '.pptx']:
            loader = UnstructuredPowerPointLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
            
        elif file_extension in ['.txt', '.log', '.json', '.xml']:
            loader = TextLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
            
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        raise ValueError(f"Failed to load document {file_path}: {str(e)}")

def transcribe_audio(audio_path):
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"Audio transcription error: {str(e)}")
        return f"[Error transcribing audio: {str(e)}]"

def transcribe_video(video_path):
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(video_path)
        return result['text']
    except Exception as e:
        print(f"Video transcription error: {str(e)}")
        return f"[Error transcribing video: {str(e)}]"

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'[^\w\s\.\,\!\?\:\;\-\"\']', ' ', text)

def split_into_sentences(text: str) -> list[str]:
    if not text:
        return []
        
    try:
        sentences = sent_tokenize(text)
        print(f"Successfully tokenized text into {len(sentences)} sentences using NLTK")
    except Exception as e:
        print(f"NLTK tokenization failed: {str(e)}, using regex fallback")
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text.split()) * 1.5) + 20  

def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
        
    words = text.split()
    words_to_keep = int(max_tokens / 1.5) - 5 
    
    if words_to_keep <= 0:
        return ""
        
    truncated = " ".join(words[:words_to_keep])
    return truncated

def create_sentence_clusters(sentences: list[str], num_clusters: int = 5) -> dict[int, list[str]]:
    if not sentences:
        return {}
        
    if len(sentences) <= 1:
        return {0: sentences}

    if len(sentences) > 500:
        print(f"Document has {len(sentences)} sentences - using direct chunking instead of clustering")
        return chunk_by_size(sentences)

    try:
        vectorizer = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.95)
        actual_clusters = min(num_clusters, max(1, len(sentences) // 2))
        
        if len(sentences) < 200:
            actual_clusters = min(actual_clusters, 5)
        
        tfidf_matrix = vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)
        
        clustered = {}
        for i, label in enumerate(labels):
            clustered.setdefault(label, []).append(sentences[i])
            
        balanced_clusters = {}
        for label, sent_list in clustered.items():
            sent_list.sort(key=len, reverse=True)
            
            current_cluster = []
            current_tokens = 0
            sub_cluster_count = 0
            
            for sentence in sent_list:
                sentence_tokens = estimate_tokens(sentence)
                
                if current_tokens + sentence_tokens > MAX_TOKENS_PER_CLUSTER and current_cluster:
                    balanced_clusters[f"{label}.{sub_cluster_count}"] = current_cluster
                    current_cluster = []
                    current_tokens = 0
                    sub_cluster_count += 1
                
                current_cluster.append(sentence)
                current_tokens += sentence_tokens
            
            if current_cluster:
                balanced_clusters[f"{label}.{sub_cluster_count}"] = current_cluster
                
        print(f"Created {len(balanced_clusters)} balanced sub-clusters")
        return balanced_clusters

    except Exception as e:
        print(f"Clustering failed: {e}")
        return chunk_by_size(sentences)

def chunk_by_size(sentences: list[str], max_tokens: int = MAX_TOKENS_PER_CLUSTER) -> dict[int, list[str]]:
    if not sentences:
        return {}
        
    chunks = {}
    current_chunk = []
    current_tokens = 0
    chunk_id = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks[chunk_id] = current_chunk
                chunk_id += 1
                current_chunk = []
                current_tokens = 0
            
            words = sentence.split()
            parts = []
            part = []
            part_tokens = 0
            
            for word in words:
                word_tokens = estimate_tokens(word)
                if part_tokens + word_tokens > max_tokens * 0.8 and part:
                    parts.append(" ".join(part))
                    part = []
                    part_tokens = 0
                
                part.append(word)
                part_tokens += word_tokens
            
            if part:
                parts.append(" ".join(part))
            
            for i, part_text in enumerate(parts):
                chunks[f"{chunk_id}.{i}"] = [part_text]
            
            chunk_id += 1
            continue
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks[chunk_id] = current_chunk
            current_chunk = []
            current_tokens = 0
            chunk_id += 1
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    if current_chunk:
        chunks[chunk_id] = current_chunk
    
    print(f"Created {len(chunks)} chunks using size-based chunking")
    return chunks

def create_llm(model_name=None):
    try:
        if not model_name:
            try:
                return ChatOpenAI(
                    model="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
                    openai_api_key="EMPTY",
                    openai_api_base=MODEL_API_BASE,
                    max_tokens=384, 
                    temperature=0
                )
            except Exception as e:
                print(f"Failed to initialize primary model: {e}")
                return ChatOpenAI(
                    model="unsloth/Qwen3-4B-unsloth-bnb-4bit",
                    openai_api_key="EMPTY",
                    openai_api_base=MODEL_API_BASE,
                    max_tokens=256, 
                    temperature=0.5
                )
        else:
            return ChatOpenAI(
                model=model_name,
                openai_api_key="EMPTY",
                openai_api_base=MODEL_API_BASE,
                max_tokens=384,
                temperature=0.5
            )
    except Exception as e:
        print(f"Error creating LLM: {e}")
        class MockLLM:
            def invoke(self, messages):
                class MockResponse:
                    content = "[Error: Unable to create language model]"
                return MockResponse()
        return MockLLM()

def summarize_text(text: str, llm) -> str:
    if not text:
        return "No text available to summarize."
        
    truncated_text = truncate_text_to_tokens(text, MAX_TOKENS_PROMPT)
    
    messages = [
        SystemMessage(content="Summarize the following text into a concise, well-structured paragraph:"),
        HumanMessage(content=truncated_text)
    ]
    
    try:
        response = llm.invoke(messages).content
        return response
    except Exception as e:
        print(f"Error in summarize_text: {e}")
        return f"[Auto-extracted summary] {text[:300]}..."

def get_cluster_summaries(clusters: dict, llm) -> list[str]:
    summaries = []
    
    for cluster_id, sentences in clusters.items():
        joined_text = " ".join(sentences)
        
        truncated_joined_text = truncate_text_to_tokens(joined_text, MAX_TOKENS_PROMPT)
        
        messages = [
            SystemMessage(content="Summarize this text into a short, concise paragraph:"),
            HumanMessage(content=truncated_joined_text)
        ]

        try:
            summary = llm.invoke(messages).content
            print(f"Successfully summarized cluster {cluster_id}")
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing cluster {cluster_id}: {e}")
            print("Falling back to extractive summarization...")
            
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
            
            summaries.append(f"[Auto-extracted summary] {fallback[:300]}...")
    
    return summaries

def create_final_summary(cluster_summaries: list[str], llm) -> str:
    if not cluster_summaries:
        return "No summary available due to processing errors."
    
    if len(cluster_summaries) > 5:
        print(f"Too many cluster summaries ({len(cluster_summaries)}). Merging...")
        chunk_size = max(1, len(cluster_summaries) // 3)
        merged_summaries = []
        
        for i in range(0, len(cluster_summaries), chunk_size):
            chunk = cluster_summaries[i:i+chunk_size]
            merged = "Section summary: " + " ".join(chunk)
            merged_summaries.append(merged)
        
        cluster_summaries = merged_summaries
    
    all_summaries = "\n\n".join([f"Section {i+1}: {s}" for i, s in enumerate(cluster_summaries)])
    
    truncated_summaries = truncate_text_to_tokens(all_summaries, MAX_TOKENS_PROMPT)

    messages = [
        SystemMessage(content="You are a skilled editor. Combine these section summaries into one cohesive, well-structured summary. Maintain key information from each section:"),
        HumanMessage(content=truncated_summaries)
    ]

    try:
        final_summary = llm.invoke(messages).content
        return final_summary
    except Exception as e:
        print(f"Error generating final summary: {e}")
        return "SUMMARY COMPILATION:\n\n" + "\n\n".join(cluster_summaries[:3])

def summarize_with_clustering(text: str, num_clusters: int = 5) -> str:
    if not text:
        return "No text available to summarize."
        
    print("Cleaning text...")
    cleaned_text = preprocess_text(text)
    
    print("Splitting into sentences...")
    sentences = split_into_sentences(cleaned_text)
    print(f"{len(sentences)} sentences found.")

    llm = create_llm()

    if len(sentences) < 10:
        print("Text is short, summarizing directly.")
        return summarize_text(cleaned_text, llm)

    print("Clustering sentences...")
    clusters = create_sentence_clusters(sentences, num_clusters)
    print(f"Created {len(clusters)} clusters/chunks.")

    print("Summarizing each cluster...")
    summaries = get_cluster_summaries(clusters, llm)

    print("Creating final summary...")
    return create_final_summary(summaries, llm)

def process_document(file_path):
    try:
        print(f"Loading document: {file_path}")
        document_text = load_document(file_path)
        
        word_count = len(document_text.split())
        print(f"Document contains approximately {word_count} words.")
        
        num_clusters = max(3, min(10, word_count // 800))
        print(f"Using {num_clusters} clusters for summarization.")
        
        final_summary = summarize_with_clustering(document_text, num_clusters)
        
        return {
            "text": document_text,
            "word_count": word_count,
            "summary": final_summary
        }
        
    except Exception as e:
        error_message = f"Error processing document: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return {"error": error_message}

if __name__ == "__main__":
    import argparse
    import sys
    
    # DEFAULT_FILE_PATH = r"D:\pyproject\summarize_ai\app\vid\SampleSuperstore.csv"
    # DEFAULT_FILE_PATH = r"D:\pyproject\summarize_ai\app\vid\jok.mp4"
    DEFAULT_FILE_PATH = r"D:\pyproject\summarize_ai\app\vid\docTest.docx"
    # DEFAULT_FILE_PATH = r"D:\pyproject\summarize_ai\app\vid\docTest.docx"
    # DEFAULT_FILE_PATH = r"D:\pyproject\summarize_ai\app\vid\pptDeploy.pptx"
    # DEFAULT_FILE_PATH = r"D:\pyproject\summarize_ai\app\vid\testPDF.pdf"
    
    # if len(sys.argv) > 1:
    #     parser = argparse.ArgumentParser(description="Process and summarize documents of various types")
    #     parser.add_argument("file_path", help="Path to the file or URL to process")
    #     parser.add_argument("--output", "-o", help="Output file path to save the summary", default=None)
    #     parser.add_argument("--model", "-m", help="Model to use for summarization", default=None)
    #     args = parser.parse_args()
    #     file_path = args.file_path
    #     output_path = args.output
    #     model_name = args.model
    # else:
    print(f"No arguments provided. Using default file path: {DEFAULT_FILE_PATH}")
    file_path = DEFAULT_FILE_PATH
    output_path = None
    model_name = None
    
    try:
        result = process_document(file_path)
        
        if "error" in result:
            print("\n=== ERROR ===\n", result["error"])
        else:
            print("\n=== DOCUMENT SUMMARY ===\n", result["summary"])
            
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("=== DOCUMENT SUMMARY ===\n\n")
                    f.write(result["summary"])
                    f.write("\n\n=== FULL TEXT ===\n\n")
                    f.write(result["text"])
                print(f"Results saved to {output_path}")
                
    except Exception as e:
        print("An unexpected error occurred:")
        print(traceback.format_exc())