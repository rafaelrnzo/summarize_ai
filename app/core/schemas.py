from pydantic import BaseModel, Field

class SummarizeRequest(BaseModel):
    file_path: str = Field(..., description="Input your file path to summarize")

class ChatRequest(BaseModel):
    user_prompt : str = Field(..., description="Input your file path to summarize")