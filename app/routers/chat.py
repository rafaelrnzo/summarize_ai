from fastapi import FastAPI, APIRouter
from core.schemas import ChatRequest
router = APIRouter(prefix="/v1/chat/file", tags=["chat"])

@router.post("/summarize")
def chat_rag(request: ChatRequest):
    print("data")