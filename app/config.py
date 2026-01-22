import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Settings:
    # FAISS
    FAISS_LOCAL_DIR = Path(
        os.getenv("FAISS_LOCAL_DIR", "./artifacts/faiss")
    )

    TOP_K = int(os.getenv("TOP_K", 3))

    # Bedrock
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
    AWS_REGION = os.getenv("S3_REGION", "us-east-1")

settings = Settings()
