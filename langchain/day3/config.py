from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")

TEMPERATURE = 0.3         # Lower for factual QA
TOP_P = 0.9
TOP_K = 40

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 3           # Number of chunks to retrieve
PDF_DIR = "docs"          # Folder for PDF documents
