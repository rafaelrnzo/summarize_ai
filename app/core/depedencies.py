from langchain_openai import ChatOpenAI
from dotenv import load_dotenv 
import os 
import traceback 

os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

MODEL_API_BASE = os.getenv("MODEL_API_BASE")
DEFAULT_MODEL = os.getenv("MODEL_MISTRAL")
TOTAL_CONTEXT_LIMIT = 1024  
MAX_RESPONSE_TOKENS = 512

def create_llm():
    try:
        return ChatOpenAI(
            model=DEFAULT_MODEL,
            openai_api_key="EMPTY",
            openai_api_base=MODEL_API_BASE,
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0
        )
    except Exception as e:
        print("Error creating LLM:")
        print(f"Type: {type(e).__name__}")
        print(f"Details: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

        class MockLLM:
            def invoke(self, messages):
                class MockResponse:
                    content = "[Error: Unable to create language model]"
                return MockResponse()
        return MockLLM()