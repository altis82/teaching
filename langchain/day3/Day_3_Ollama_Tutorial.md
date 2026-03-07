# Day 3 – RAG System: Question-Answering from Documents (Intermediate) ⭐

> **This is the most important lab** – RAG is fundamental to production LLM applications.

## 🎯 Learning Objectives

Build a Retrieval-Augmented Generation (RAG) system that answers questions based on your own PDF documents, using a local Ollama model for both the LLM and embeddings.

## 🧠 What You Will Learn

- **Embeddings**: Converting text into vector representations that capture semantic meaning
- **Vector Store (FAISS)**: Efficient storage and similarity search of document embeddings
- **Retriever**: Selecting the most relevant document chunks for a given query
- **RAG Pipeline**: Combining retrieval with generation for accurate, grounded answers
- **Document Processing**: Loading PDFs, splitting into chunks, and preparing for retrieval

## 🛠 Technical Requirements

### 1) Ollama Setup (same as Day 2)

Make sure Ollama is running and you have pulled the required models:

```bash
ollama serve                     # in one terminal
ollama pull llama3.2:3b          # LLM model
ollama pull qwen3-embedding:8b   # embedding model
```

Verify both are available:

```bash
ollama list
```

### 2) Install Python Dependencies

```bash
cd /home/system/teaching/langchain/day3
pip install -r requirements.txt
```

Key components:
| Package | Purpose |
|---|---|
| `langchain` | Core framework |
| `langchain-ollama` | Ollama LLM + Embeddings wrappers |
| `faiss-cpu` | Vector similarity search |
| `pypdf` | PDF text extraction |
| `chromadb` | Alternative vector DB (optional) |
| `sentence-transformers` | Alternative embedding models |

### 3) Prepare Documents

Create a `docs/` folder and place one or more PDF files inside:

```bash
mkdir -p docs
# Copy any PDF here, e.g.:
# cp ~/reports/annual_report.pdf docs/
# cp ~/papers/ai_research.pdf docs/
```

> **Tip**: For testing, use any freely available PDF – a Wikipedia article exported to PDF, an open research paper from [arXiv](https://arxiv.org), etc.

---

## 🚀 Practical Exercise: Document Q&A System

### Architecture Overview

```
PDF Documents
    ↓
[1. Load] → PyPDFLoader reads each page
    ↓
[2. Split] → RecursiveCharacterTextSplitter creates overlapping chunks
    ↓
[3. Embed] → OllamaEmbeddings converts chunks to vectors
    ↓
[4. Store] → FAISS indexes vectors for fast similarity search
    ↓  (saved to disk – only run once per document set)
    ↓
User Question
    ↓
[5. Retrieve] → FAISS finds top-K similar chunks
    ↓
[6. Generate] → ChatOllama answers using retrieved context
    ↓
Answer with source citations
```

---

### Step 1: Load PDF Documents

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_documents(pdf_dir="docs"):
    loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    print(f"[load] {len(docs)} pages loaded")
    return docs
```

**What happens here:**
- `DirectoryLoader` scans the `docs/` folder for all `.pdf` files.
- `PyPDFLoader` extracts text from each page → each page becomes one `Document` object.
- Each document carries `metadata` with `source` (filename) and `page` number.

**Try it:**
```python
docs = load_documents()
print(docs[0].page_content[:200])   # first 200 chars of page 1
print(docs[0].metadata)             # {'source': 'docs/file.pdf', 'page': 0}
```

---

### Step 2: Split Documents into Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # max characters per chunk
        chunk_overlap=200,     # overlap between adjacent chunks
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[split] {len(chunks)} chunks")
    return chunks
```

**Why chunking matters:**

| Setting | Too Small (200) | Good (1000) | Too Large (3000) |
|---|---|---|---|
| Context per chunk | Fragmented sentences | Full paragraphs | Multiple topics mixed |
| Retrieval precision | High noise | Balanced | Low precision |
| Token usage | Many small calls | Efficient | May exceed context window |

**Why overlap?**
- Without overlap: a sentence split across two chunks loses meaning in both.
- With 200-char overlap: boundary sentences appear in both adjacent chunks → no information lost.

**Visual example:**
```
Original text: [A A A A A|B B B B B|C C C C C]

No overlap:    [A A A A A] [B B B B B] [C C C C C]
                     ↑ sentence cut here is lost

With overlap:  [A A A A A B B] [A B B B B B C C] [B C C C C C]
                  overlap ^^^     overlap ^^^
```

---

### Step 3: Create Embeddings and Vector Store

```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings():
    return OllamaEmbeddings(
        base_url="http://localhost:11434",
        model="qwen3-embedding:8b",
    )

def build_vectorstore(chunks):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")    # persist to disk
    return vectorstore
```

**What happens here:**
1. Each chunk's text → sent to Ollama embedding model → returns a high-dimensional vector (e.g., 1024 floats).
2. FAISS indexes all vectors for fast nearest-neighbor search.
3. `save_local()` writes the index to disk so you don't re-embed every time.

**Embedding intuition:**
```
"AI is transforming banking"  →  [0.12, -0.45, 0.78, ...]  (1024 dims)
"Machine learning in finance" →  [0.11, -0.43, 0.80, ...]  (similar!)
"Recipe for chocolate cake"   →  [0.89,  0.22, -0.15, ...] (very different)
```

Similar meanings → vectors close together → FAISS finds them fast.

---

### Step 4: Build the RAG Chain

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},         # top 3 most relevant chunks
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer based on the provided context. "
         "If the context is insufficient, say so. Cite sources.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ])

    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        temperature=0.3,
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source','?')}, "
            f"page {d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever
```

**Chain flow explained:**

```
User question: "What was total revenue?"
    │
    ├─→ retriever | format_docs
    │       │
    │       └─→ FAISS finds 3 most similar chunks
    │           └─→ format_docs joins them with source citations
    │               → becomes {context}
    │
    └─→ RunnablePassthrough()
            └─→ question string passes through unchanged
                → becomes {question}
    │
    ↓
    prompt.format(context=..., question=...)
    ↓
    ChatOllama generates answer grounded in context
    ↓
    StrOutputParser extracts text
    ↓
    "Total revenue was $X million (Source: report.pdf, page 5)"
```

---

### Step 5: Run the Full Pipeline

```python
# First time: ingest documents
docs = load_documents()
chunks = split_documents(docs)
vectorstore = build_vectorstore(chunks)

# Build chain
chain, retriever = build_rag_chain(vectorstore)

# Ask questions
answer = chain.invoke("What are the main findings?")
print(answer)
```

### Step 6: Interactive Q&A Loop

The complete `main.py` provides an interactive CLI:

```bash
cd /home/system/teaching/langchain/day3
python main.py
```

```
=== RAG Q&A System ===
Ask questions about your documents.
Commands:  'ingest' = re-index docs  |  'exit' = quit

you > What is the total revenue?

[retrieved 3 chunks]
  1. docs/report.pdf (page 5) – Total revenue for FY2025 reached $2.3 billion...
  2. docs/report.pdf (page 12) – Revenue breakdown by segment shows...
  3. docs/report.pdf (page 3) – Executive summary: record revenue...

bot > Based on the annual report, total revenue for FY2025 was $2.3 billion.
      (Source: report.pdf, page 5)
```

---

## 🔧 Common Troubleshooting

| Problem | Solution |
|---|---|
| `No PDFs found` | Add PDF files to `docs/` folder |
| `Connection refused` | Run `ollama serve` in another terminal |
| `Model not found` | Run `ollama pull llama3.2:3b` and `ollama pull qwen3-embedding:8b` |
| Slow embedding | Embedding 100+ pages takes time on CPU. Be patient or use smaller model |
| Bad retrieval quality | Try adjusting `chunk_size` (500, 1000, 2000) and `k` (3, 5, 7) |
| `FAISS import error` | `pip install faiss-cpu` (not `faiss-gpu` unless you have CUDA) |

---

## ⭐ Advanced Extensions

### A) Experiment with Chunk Sizes

Run the same questions with different chunk settings and compare answer quality:

```python
experiments = [
    {"chunk_size": 500,  "chunk_overlap": 100, "k": 5},
    {"chunk_size": 1000, "chunk_overlap": 200, "k": 3},
    {"chunk_size": 2000, "chunk_overlap": 300, "k": 2},
]
```

Record: retrieval relevance, answer accuracy, token usage, latency.

### B) Hybrid Search (Vector + Keyword)

FAISS does pure vector similarity. For exact keyword matches (e.g., "Q4 2025"), add BM25:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(chunks, k=3)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

hybrid = EnsembleRetriever(
    retrievers=[bm25, faiss_retriever],
    weights=[0.4, 0.6],    # 40% keyword, 60% semantic
)
```

### C) Metadata Filtering

Filter retrieved docs by source file or page range:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source": "docs/q4_report.pdf"},
    }
)
```

### D) Add Token Logging

Reuse the pattern from Day 2:

```python
meta = getattr(response, "response_metadata", {}) or {}
prompt_tokens = int(meta.get("prompt_eval_count", 0))
completion_tokens = int(meta.get("eval_count", 0))
print(f"[tokens] prompt={prompt_tokens} completion={completion_tokens}")
```

### E) Use ChromaDB Instead of FAISS

ChromaDB supports metadata filtering out of the box:

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    chunks, embeddings,
    collection_name="day3_docs",
    persist_directory="./chroma_db",
)
```

---

## 📝 Key Takeaways

1. **RAG = Retrieval + Generation** – the LLM only sees relevant chunks, not the entire document.
2. **Chunk size is critical** – too small loses context, too large adds noise.
3. **Embeddings capture meaning** – "revenue" and "income" are close in vector space.
4. **FAISS is fast** – millions of vectors searched in milliseconds.
5. **Always cite sources** – production RAG should show where answers come from.
6. **Persist your index** – re-embedding is expensive; save to disk and reload.
