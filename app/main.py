from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.schemas import SummarizeRequest
import traceback
from routers import summarize, chat, scan 
import nltk 

app = FastAPI(title="AI Summarize Data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(summarize.router)
app.include_router(scan.router)
# app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "AI Summarize Data Archive"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.100.130", port=8004)
