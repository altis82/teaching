from dotenv import load_dotenv
import os
from pathlib import Path

env_path=Path(__file__).parent / ".env"
load_dotenv(env_path)    

OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL=os.getenv("OLLAMA_MODEL")
OLLAMA_API_KEY=os.getenv("OLLAMA_API_KEY")

TEMPERATURE = 0.7        # 0 = deterministic, 1.0 = creative
TOP_P = 0.9             # Controls diversity
TOP_K = 40              # Top-k sampling

# Application settings
NUM_TITLES = 5
TIMEOUT_SECONDS = 60

REDIS_URL=os.getenv("REDIS_URL")

SUMMARIZE_THRESHOLD=12