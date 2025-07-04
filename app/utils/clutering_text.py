import re 
from typing import List, Dict, Any
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans  # ✅ ganti KMeans ke MiniBatchKMeans
from dotenv import load_dotenv
import os

load_dotenv()

MAX_TOKENS_PER_CLUSTER = int(os.getenv("MAX_TOKENS_PER_CLUSTER", "500"))
CLUSTER_WORD_RATIO = int(os.getenv("CLUSTER_WORD_RATIO", "600"))

def preprocess_text(text: str) -> str:
    """Preprocess text for clustering"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'[^\w\s\.\,\!\?\:\;\-\"\']', ' ', text)

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences with fallback"""
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
    """Estimate token count for text"""
    if not text:
        return 0
    return int(len(text.split()) * 1.5) + 20  

def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit"""
    if estimate_tokens(text) <= max_tokens:
        return text
        
    words = text.split()
    words_to_keep = int(max_tokens / 1.5) - 5 
    
    if words_to_keep <= 0:
        return ""
        
    truncated = " ".join(words[:words_to_keep])
    return truncated

def create_sentence_clusters(sentences: List[str], num_clusters: int = 5) -> Dict[Any, List[str]]:
    """Create sentence clusters using TF-IDF and MiniBatchKMeans for better speed"""
    if not sentences:
        return {}
        
    if len(sentences) <= 1:
        return {0: sentences}

    if len(sentences) > 500:
        print(f"Document has {len(sentences)} sentences - using direct chunking instead of clustering")
        return chunk_by_size(sentences)

    try:
        # Configure vectorizer
        max_features = min(500, len(sentences))  # Turunkan max_features untuk speed
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=1,
            max_df=0.95,
            stop_words='english',
            ngram_range=(1, 1)  # 1-gram untuk speed
        )

        actual_clusters = min(num_clusters, max(1, len(sentences) // 2))
        if len(sentences) < 200:
            actual_clusters = min(actual_clusters, 5)

        tfidf_matrix = vectorizer.fit_transform(sentences)

        # MiniBatchKMeans untuk kecepatan
        kmeans = MiniBatchKMeans(
            n_clusters=actual_clusters,
            random_state=42,
            batch_size=100,
            max_iter=100
        )

        labels = kmeans.fit_predict(tfidf_matrix)

        clustered = {}
        for i, label in enumerate(labels):
            clustered.setdefault(label, []).append(sentences[i])

        balanced_clusters = balance_clusters_by_tokens(clustered)

        print(f"Created {len(balanced_clusters)} balanced clusters (MiniBatchKMeans)")
        return balanced_clusters

    except Exception as e:
        print(f"Clustering failed: {e}")
        return chunk_by_size(sentences)

def balance_clusters_by_tokens(clustered: Dict[int, List[str]]) -> Dict[str, List[str]]:
    """Balance clusters to ensure they don't exceed token limits"""
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
    
    return balanced_clusters

def chunk_by_size(sentences: List[str], max_tokens: int = None) -> Dict[int, List[str]]:
    """Chunk sentences by size when clustering is not feasible"""
    if max_tokens is None:
        max_tokens = MAX_TOKENS_PER_CLUSTER
        
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
            
            parts = split_long_sentence(sentence, max_tokens)
            
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

def split_long_sentence(sentence: str, max_tokens: int) -> List[str]:
    """Split a long sentence into smaller parts"""
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
    
    return parts
