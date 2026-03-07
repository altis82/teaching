"""
Day 3 – RAG System: Question-Answering from Documents
Uses local Ollama for both LLM and embeddings.
"""

from pathlib import Path




from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL,
    TEMPERATURE, TOP_P, TOP_K,
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K, PDF_DIR,
)

VECTORSTORE_PATH = "vectorstore"




if __name__ == "__main__":
    main()
