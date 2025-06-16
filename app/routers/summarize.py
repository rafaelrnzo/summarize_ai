from fastapi import FastAPI, APIRouter, HTTPException, status
from core.schemas import SummarizeRequest
from services.ServicesSummarize import process_document
import logging
from typing import Dict, Any
import time
import re
from utils.split_sentence import split_into_sentences

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/file", tags=["summarize"])


@router.post(
    "/summarize",
    response_model=Dict[str, Any],
    summary="Summarize document content",
    description="Process and summarize the content of a document at the given file path."
)
async def file_summarize(request: SummarizeRequest) -> Dict[str, Any]:
    file_path = request.file_path
    start_time = time.time()

    try:
        result = process_document(file_path)

        if "error" in result:
            logger.error(f"Error processing document: {result['error']}")
            return {
                "status": "error",
                "message": result["error"],
                "process_time": round(time.time() - start_time, 4)
            }

        if "summary" in result and isinstance(result["summary"], str):
            result["summary"] = result["summary"].strip()

        all_text = ""
        if "scripts" in result:
            if isinstance(result["scripts"], list):
                if all(isinstance(item, dict) and "text" in item for item in result["scripts"]):
                    all_text = " ".join(item["text"] for item in result["scripts"])
                elif all(isinstance(item, str) for item in result["scripts"]):
                    all_text = " ".join(result["scripts"])

        scripts_clean = split_into_sentences(all_text)

        return {
            "status": "success",
            "file_path": file_path,
            "response": {
                "summary": result.get("summary", ""),
                "scripts": scripts_clean
            },
            "process_time": round(time.time() - start_time, 4)
        }

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {
            "status": "error",
            "message": f"File not found: {file_path}",
            "process_time": round(time.time() - start_time, 4)
        }
    except PermissionError:
        logger.error(f"Permission denied when accessing file: {file_path}")
        return {
            "status": "error",
            "message": f"Permission denied when accessing file: {file_path}",
            "process_time": round(time.time() - start_time, 4)
        }
    except Exception as e:
        logger.exception(f"An unexpected error occurred while processing {file_path}")
        return {
            "status": "error",
            "message": "An internal server error occurred.",
            "error_type": type(e).__name__,
            "process_time": round(time.time() - start_time, 4)
        }
