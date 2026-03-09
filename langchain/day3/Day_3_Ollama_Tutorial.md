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

## 💬 Adding Chat History (Session ID)

By default, the RAG chain is **stateless** — each question is independent. To make the system remember previous questions and answers within a conversation, we wrap the chain with `RunnableWithMessageHistory`.

### Why Session ID?

- Each `session_id` has its own isolated chat history.
- User A and User B can chat simultaneously without interfering.
- Resetting a session is as simple as using a new `session_id`.

### Step-by-step Changes

#### 1. Additional Imports

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
```

#### 2. Update the Prompt — Add History Placeholder

The key change: insert `MessagesPlaceholder("chat_history")` between the system message and the human message. This is where LangChain will inject past messages.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer based on the provided context. "
     "If the context is insufficient, say so. Cite sources.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),   # ← NEW
    ("human", "{question}"),
])
```

**Before (stateless):**
```
[system] Context: ...
[human]  What is the revenue?
```

**After (with history):**
```
[system]   Context: ...
[human]    What is the revenue?          ← from history
[assistant] Revenue was $2.3B.           ← from history
[human]    Break it down by segment.     ← current question
```

#### 3. Modify the Chain Input Format

The chain now receives a **dict** instead of a plain string, because we need separate keys for `question` and `chat_history`:

```python
chain = (
    {
        "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
        "question": RunnableLambda(lambda x: x["question"]),
        "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

**Flow comparison:**

| | Stateless Chain | With History |
|---|---|---|
| Input | `"What is revenue?"` (string) | `{"question": "What is revenue?"}` (dict) |
| Context key | `retriever` receives string directly | `RunnableLambda` extracts `x["question"]` → retriever |
| Question key | `RunnablePassthrough()` passes string | `RunnableLambda` extracts `x["question"]` |
| History key | _(none)_ | `RunnableLambda` extracts `x["chat_history"]` |

#### 4. Create Session Store and Wrap the Chain

```python
# In-memory store: each session_id → its own ChatMessageHistory
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",       # which key holds the user's question
    history_messages_key="chat_history", # which key to inject history into
)
```

**What `RunnableWithMessageHistory` does automatically:**
1. Before calling the chain: loads history from `store[session_id]` → injects into `chat_history`.
2. After the chain returns: saves the (human question, AI answer) pair back to `store[session_id]`.

#### 5. Invoke with Session ID

```python
# Same session → model remembers context
answer = chain_with_history.invoke(
    {"question": "What is the total revenue?"},
    config={"configurable": {"session_id": "user_alice"}}
)

# Follow-up in same session → model knows "it" = revenue
answer = chain_with_history.invoke(
    {"question": "Break it down by segment."},
    config={"configurable": {"session_id": "user_alice"}}
)

# Different session → fresh start, no memory of Alice's questions
answer = chain_with_history.invoke(
    {"question": "Who are the authors?"},
    config={"configurable": {"session_id": "user_bob"}}
)
```

#### 6. Interactive Loop with Session ID

```python
def main():
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    chain_with_history, retriever = build_rag_chain(vectorstore)

    session_id = "default"
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        answer = chain_with_history.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"Bot: {answer}\n")
```

### Memory Store Options

The in-memory `dict` store works for development but is lost on restart. For production:

| Store | Persistence | Use Case |
|---|---|---|
| `ChatMessageHistory()` (dict) | In-memory only | Development, testing |
| `RedisChatMessageHistory` | Redis server | Multi-process, scalable |
| `SQLChatMessageHistory` | SQLite/PostgreSQL | Persistent, queryable |
| `FileChatMessageHistory` | Local JSON files | Simple persistence |

Example with Redis:
```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id: str):
    return RedisChatMessageHistory(session_id, url="redis://localhost:6379")
```

---

## 🧠 Smarter Retrieval: When the Answer Exists but the Retriever Misses It

### The Problem

Ask: *"What is the title of the document?"*

The retriever returns chunks from **pages 10–11** (the references section, full of other paper titles) instead of **page 0** where the actual document title is.

**Why?** Pure similarity search matches the word "title" semantically with reference entries that contain many paper titles. The actual title on page 0 doesn't contain the word "title" — it just *is* the title.

```
Query: "What is the title?"

Similarity search returns:
  ✗ page 10: "[3] A. Smith, 'Deep Learning for Network...', IEEE Trans..."  ← has "title-like" text
  ✗ page 11: "[7] B. Jones, 'UAV-Assisted Emergency...', ACM..."          ← has "title-like" text
  ✗ page 10: "[1] C. Lee, 'VNF Orchestration in 5G...', JSAC..."          ← has "title-like" text

What we actually needed:
  ✓ page 0: "Dynamic VNF Orchestration for UAV-aided Emergency Networks..."  ← THE title
```

This is a **fundamental limitation of naive similarity search**. Here are strategies to fix it:

---

### Strategy 1: Multi-Query Retriever (Recommended)

Instead of searching once with the user's exact question, use the LLM to **rephrase the question into multiple search queries** → run all of them → merge results.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOllama(base_url="http://localhost:11434", model="llama3.2:3b"),
)
```

**What happens internally:**

```
User question: "What is the title of the document?"

LLM generates 3 variants:
  1. "What is the name of this research paper?"
  2. "What is the main topic discussed in the first page?"
  3. "What is the heading of this academic paper?"

Each query retrieves top-3 chunks → union of all results → deduplicate
→ Now page 0 chunks are very likely to appear!
```

**Why it works:** The reformulated queries approach the same information from different angles. "Main topic discussed in the first page" will semantically match the actual title chunk.

Use it in the chain:
```python
chain = (
    {"context": multi_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

### Strategy 2: MMR — Maximum Marginal Relevance

Instead of returning the top-3 *most similar* chunks (which may all be from the references section), MMR balances **relevance** and **diversity**.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",          # ← change from "similarity" to "mmr"
    search_kwargs={
        "k": 5,                 # return 5 chunks
        "fetch_k": 20,          # consider top 20 candidates
        "lambda_mult": 0.5,     # 0 = max diversity, 1 = max relevance
    }
)
```

**How MMR works:**

```
Step 1: Fetch top 20 most similar chunks (fetch_k=20)
Step 2: Pick the most relevant one → [ref page 10]
Step 3: Pick next most relevant that is DIFFERENT from step 2 → [page 0]  ✓
Step 4: Pick next most relevant that is different from 2&3 → [page 5]
...until k=5 chunks selected
```

**Why it helps:** Even if references rank highest, MMR forces diversity → chunks from other parts of the document get included.

---

### Strategy 3: Contextual Compression — Re-rank After Retrieval

Retrieve more chunks than needed, then use the LLM to **filter and re-rank** them based on the actual question.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter

# Fetch many candidates
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# LLM filters out irrelevant ones
compressor = LLMChainFilter.from_llm(
    ChatOllama(base_url="http://localhost:11434", model="llama3.2:3b")
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
```

**Flow:**
```
Query: "What is the title?"
  → base_retriever returns 10 chunks
  → LLM evaluates each: "Is this chunk relevant to finding the document title?"
     ✗ Reference list [3] A. Smith...  → NO, this is a citation
     ✗ Reference list [7] B. Jones...  → NO, this is a citation
     ✓ "Dynamic VNF Orchestration..."  → YES, this is the document title
  → Only relevant chunks pass through
```

**Trade-off:** More accurate, but slower (LLM called for each candidate chunk).

---

### Strategy 4: Enrich Chunks with Contextual Metadata

The problem: chunks are just raw text without context about *where* they are in the document. Add structural metadata before embedding.

```python
def enrich_chunks(chunks):
    """Add contextual prefix to each chunk before embedding."""
    for chunk in chunks:
        page = chunk.metadata.get("page", 0)
        source = chunk.metadata.get("source", "unknown")

        # Add position context
        if page == 0:
            prefix = f"[DOCUMENT FRONT PAGE - likely contains title, authors, abstract] "
        elif page <= 2:
            prefix = f"[INTRODUCTION SECTION] "
        else:
            prefix = f"[Page {page}] "

        chunk.page_content = prefix + chunk.page_content

    return chunks

# Use before building vectorstore:
chunks = split_documents(docs)
chunks = enrich_chunks(chunks)       # ← add this
vectorstore = build_vectorstore(chunks)
```

**Why it works:** Now when searching for "title", the chunk from page 0 has `[DOCUMENT FRONT PAGE - likely contains title...]` in its text → much higher similarity score.

---

### Strategy 5: Combine Multiple Strategies

For production RAG, combine approaches:

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

# Strategy 1: Multi-query + MMR
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 15, "lambda_mult": 0.5}
)
multi_query = MultiQueryRetriever.from_llm(
    retriever=mmr_retriever, llm=llm
)

# Strategy 2: BM25 keyword search
from langchain_community.retrievers import BM25Retriever
bm25 = BM25Retriever.from_documents(chunks, k=3)

# Combine both
ensemble = EnsembleRetriever(
    retrievers=[multi_query, bm25],
    weights=[0.6, 0.4],
)
```

### Quick Comparison

| Strategy | Accuracy | Speed | Complexity | Best For |
|---|---|---|---|---|
| **Multi-Query** | ⭐⭐⭐⭐ | Medium | Low | General improvement |
| **MMR** | ⭐⭐⭐ | Fast | Minimal | Diverse results |
| **Contextual Compression** | ⭐⭐⭐⭐⭐ | Slow | Medium | Precision-critical |
| **Enrich Metadata** | ⭐⭐⭐ | Fast | Low | Structured documents |
| **Ensemble (combined)** | ⭐⭐⭐⭐⭐ | Slow | High | Production systems |

**Recommendation:** Start with **MMR** (one-line change) + **Multi-Query** for significant improvement with minimal code changes.

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
