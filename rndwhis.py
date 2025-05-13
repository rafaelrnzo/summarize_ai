import whisper
import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

VIDEO_PATH = "vid/1jam.mp4"
MODEL_API_BASE = "http://192.168.100.3:8000/v1"

def ensure_nltk_resources():
    for resource in ['punkt']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

ensure_nltk_resources()

def transcribe_video(video_path: str) -> str:
    model = whisper.load_model("tiny")
    result = model.transcribe(video_path)
    return result['text']

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'[^\w\s\.\,\!\?\:\;\-\"\']', ' ', text)

def split_into_sentences(text: str) -> list[str]:
    try:
        sentences = sent_tokenize(text)
    except Exception:
        print("NLTK tokenization failed, using regex fallback")
        sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def create_sentence_clusters(sentences: list[str], num_clusters: int = 5) -> dict[int, list[str]]:
    if len(sentences) <= 1:
        return {0: sentences}

    vectorizer = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.95)
    actual_clusters = min(num_clusters, max(1, len(sentences) // 2))

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)
        
        clustered = {}
        for i, label in enumerate(labels):
            clustered.setdefault(label, []).append(sentences[i])
        return clustered

    except Exception as e:
        print(f"Clustering failed: {e}")
        return {0: sentences}

def summarize_text(text: str, llm) -> str:
    messages = [
        SystemMessage(content="Summarize the following text into a concise, well-structured paragraph."),
        HumanMessage(content=text)
    ]
    return llm.invoke(messages).content

def get_cluster_summaries(clusters: dict[int, list[str]], llm) -> list[str]:
    summaries = []
    for cluster_id, sentences in clusters.items():
        joined_text = " ".join(sentences)
        truncated = " ".join(joined_text.split()[:800])
        
        messages = [
            SystemMessage(content="Summarize this cluster into one clear paragraph."),
            HumanMessage(content=truncated)
        ]

        try:
            summaries.append(llm.invoke(messages).content)
        except Exception as e:
            print(f"Error summarizing cluster {cluster_id}: {e}")
            summaries.append(truncated[:500] + "...")
    return summaries

def create_final_summary(cluster_summaries: list[str], llm) -> str:
    all_summaries = "\n\n".join([f"Section {i+1}: {s}" for i, s in enumerate(cluster_summaries)])

    messages = [
        SystemMessage(content="You are a skilled editor. Combine these section summaries into one final paragraph."),
        HumanMessage(content=all_summaries)
    ]

    try:
        return llm.invoke(messages).content
    except Exception as e:
        print(f"Error generating final summary: {e}")
        return all_summaries

def summarize_with_clustering(text: str, num_clusters: int = 5) -> str:
    print("Cleaning text...")
    cleaned_text = preprocess_text(text)
    
    print("Splitting into sentences...")
    sentences = split_into_sentences(cleaned_text)
    print(f"{len(sentences)} sentences found.")

    llm = ChatOpenAI(
        model="unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
        openai_api_key="EMPTY",
        openai_api_base=MODEL_API_BASE,
        max_tokens=1024,
        temperature=0.7
    )

    if len(sentences) < 10:
        print("Text is short, summarizing directly.")
        return summarize_text(text, llm)

    print("Clustering sentences...")
    clusters = create_sentence_clusters(sentences, num_clusters)

    print("Summarizing each cluster...")
    summaries = get_cluster_summaries(clusters, llm)

    print("Creating final summary...")
    return create_final_summary(summaries, llm)

if __name__ == "__main__":
    try:
        print("Transcribing video...")
        transcript = transcribe_video(VIDEO_PATH)
        word_count = len(transcript.split())
        print(f"Transcript contains {word_count} words.")

        num_clusters = max(3, min(8, word_count // 1000 + 1))
        print(f"Using {num_clusters} clusters.")

        final_summary = summarize_with_clustering(transcript, num_clusters)
        print("\n=== FINAL SUMMARY ===\n", final_summary)

    except Exception as e:
        import traceback
        print("An error occurred:")
        print(traceback.format_exc())
