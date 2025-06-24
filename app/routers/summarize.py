import os
import time
import logging
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from core.schemas import SummarizeRequest
from services.ServicesSummarize import process_document
from utils.split_sentence import split_into_sentences

BASE_DIR = Path(os.getenv("BASE_DIR") ).resolve()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/file", tags=["summarize"])

@router.post("/summarize", response_model=Dict[str, Any])
async def file_summarize(request: SummarizeRequest) -> Dict[str, Any]:
    start_time = time.time()

    try:
        # Sanitize and resolve path
        requested_path = (BASE_DIR / request.file_path.strip("/\\")).resolve()

        # Prevent path traversal
        if not str(requested_path).startswith(str(BASE_DIR)):
            raise HTTPException(status_code=400, detail="Invalid file path. Access denied.")

        if not requested_path.exists():
            raise FileNotFoundError(f"File not found: {requested_path}")

        result = process_document(str(requested_path))

        if "error" in result:
            logger.error(f"Processing error: {result['error']}")
            return {
                "status": "error",
                "message": result["error"],
                "process_time": round(time.time() - start_time, 4)
            }

        summary = result.get("summary", "")
        if isinstance(summary, str):
            summary = summary.strip()

        scripts_raw = result.get("scripts", [])
        all_text = ""
        if isinstance(scripts_raw, list):
            if all(isinstance(item, dict) and "text" in item for item in scripts_raw):
                all_text = " ".join(item["text"] for item in scripts_raw)
            elif all(isinstance(item, str) for item in scripts_raw):
                all_text = " ".join(scripts_raw)

        scripts_clean = split_into_sentences(all_text)

        return {
            "status": "success",
            "file_path": request.file_path,
            "response": {
                "summary": summary,
                "scripts": scripts_clean
            },
            "process_time": round(time.time() - start_time, 4)
        }

    except FileNotFoundError as e:
        logger.warning(str(e))
        return {
            "status": "error",
            "message": str(e),
            "process_time": round(time.time() - start_time, 4)
        }

    except PermissionError as e:
        logger.warning(f"Permission error: {str(e)}")
        return {
            "status": "error",
            "message": f"Permission denied when accessing file: {request.file_path}",
            "process_time": round(time.time() - start_time, 4)
        }

    except Exception as e:
        logger.exception("Unexpected error")
        return {
            "status": "error",
            "message": "Internal server error.",
            "error_type": type(e).__name__,
            "process_time": round(time.time() - start_time, 4)
        }
