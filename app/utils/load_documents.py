from utils.file_type import get_file_type
import os
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
from utils.transcribe import transcribe_audio, transcribe_video


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
    
