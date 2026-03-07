from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

TEMPERATURE = 0.0         # Deterministic for tool-calling agents
TOP_P = 0.9
TOP_K = 40

CSV_DIR = "data"          # Folder for CSV sample data
