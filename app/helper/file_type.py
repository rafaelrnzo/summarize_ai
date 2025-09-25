from pathlib import Path
import mimetypes

def get_file_type(file_path):
    file_extension = Path(file_path).suffix.lower()
    mime_type, _ = mimetypes.guess_type(file_path)
    
    return file_extension, mime_type