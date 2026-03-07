
# Lab 2 – Stateful Chatbot with Memory Management (Beginner → Intermediate)

## 🎯 Learning Objectives

Build a chatbot that maintains conversation history and understands context to provide personalized, coherent responses across multiple turns.

## 🧠 What You Will Learn

- **ChatMessageHistory**: How to store and retrieve conversation history
- **Runnable Interface**: Working with modern LangChain chains and their composition
- **Stateful Interactions**: Managing context across multiple user-bot exchanges
- **Message History Management**: Structuring chat history for LLM consumption
- **Memory Types**: Different strategies for managing conversation context

## 🛠 Technical Requirements

### 1) Install Ollama

Linux/macOS:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify install:
```bash
ollama --version
```

Start Ollama service (if not already running):
```bash
ollama serve
```

In another terminal, pull a model:
```bash
ollama pull llama3.2:3b
```

Quick sanity test:
```bash
ollama run llama3.2:3b "Say hello in one sentence."
```

### 2) Install Python Dependencies

Install dependencies:
```bash
pip install -U langchain langchain-ollama
pip install -U langchain-community
```

Key components:
- `ChatMessageHistory`: Stores session messages in memory
- `ChatPromptTemplate`: Specialized template for chat interactions
- `RunnableWithMessageHistory`: Enables stateful chain execution
- `ChatOllama`: LangChain chat model wrapper for local Ollama models

## 🚀 Practical Exercise: Personal Memory Chatbot

### Objective
Create a chatbot that remembers user information and preferences to provide personalized responses.

### Requirements

**1️⃣ Core Features**

Your chatbot must:
- Remember the user's name after they introduce themselves
- Store and recall user preferences and interests
- Provide contextual responses that reference previous information
- Maintain conversation flow naturally

Example conversation:
```
User: My name is Chuan.
Bot: Nice to meet you, Chuan! What can I help you with today?

User: I'm interested in artificial intelligence and machine learning.
Bot: That's fascinating! AI and ML are rapidly evolving fields.

User: What are my interests again?
Bot: You mentioned you're interested in artificial intelligence and machine learning.

User: Tell me a joke about my interests.
Bot: Why did the AI go to school? To improve its learning rate! I know you're passionate about AI and ML, so this should resonate with you.
```

**2️⃣ Implementation Steps**

1. Initialize a session history store (for example `ChatMessageHistory`)
2. Create a `ChatPromptTemplate` with system prompt that acknowledges context
3. Wrap your chain with `RunnableWithMessageHistory`
4. Implement input validation
5. Test conversation flows with multiple exchanges

**3️⃣ Sample Code Structure**

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE, TOP_K, TOP_P

# In-memory history store keyed by session id.
store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def build_chatbot() -> RunnableWithMessageHistory:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that remembers user information."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL or "llama3.2:3b",
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    chain = prompt | llm

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def main() -> None:
    chatbot = build_chatbot()
    session_id = "user-1"

    print("Interactive Ollama chat started.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("you > ").strip()

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("bye")
            break

        response = chatbot.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"bot > {response.content}\n")


if __name__ == "__main__":
    main()

```

### 4) Common Ollama Troubleshooting

- `connection refused` or timeout:
  - Ensure `ollama serve` is running.
  - Default endpoint is `http://localhost:11434`.
- `model not found`:
  - Run `ollama pull llama3.2:3b` (or your selected model).
- Slow responses:
  - Use a smaller model (for example `llama3.2:1b`) or reduce concurrency.

## ⭐ Advanced Extensions

**Memory Optimization**

Goal: measure how much context costs in tokens and how it grows over time.

### A) Capture token usage per turn

Ollama returns token metadata that you can log after each response.

Typical metadata fields:
- `prompt_eval_count`: tokens used for prompt/history
- `eval_count`: tokens used for generated output

Example snippet to add after each chatbot response:

```python
meta = getattr(response, "response_metadata", {}) or {}
prompt_tokens = int(meta.get("prompt_eval_count", 0))
completion_tokens = int(meta.get("eval_count", 0))
total_tokens = prompt_tokens + completion_tokens

print(f"[tokens] prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}")
```

### B) Compare scenarios

Run these scenario tests and record average token usage over 10 turns each:
- Scenario 1: no memory (fresh session id each turn)
- Scenario 2: full history memory (single session id)
- Scenario 3: summarized history (periodic compression)

Recommended comparison table:

| Scenario | Avg Prompt Tokens | Avg Completion Tokens | Avg Total Tokens | Notes |
|---|---:|---:|---:|---|
| No memory |  |  |  | Lowest context cost |
| Full memory |  |  |  | Best recall, highest growth |
| Summarized |  |  |  | Balanced recall/cost |

### C) Estimate cost impact

For local Ollama, direct cost is usually 0 USD, but token trends still matter for speed/latency.
If you want cloud-style estimates:

```python
input_price_per_1k = 0.002   # example only
output_price_per_1k = 0.006  # example only

est_cost = (prompt_tokens / 1000) * input_price_per_1k + (completion_tokens / 1000) * output_price_per_1k
```

Use this to show how long chats increase prompt tokens and response time.

**Alternative Memory Strategies**

Your current pattern uses `RunnableWithMessageHistory`, which is modern and flexible.
For long chats, prefer **periodic summarization** over storing full raw history forever.

### A) Sliding window memory

Keep only the last N turns (for example, 6-10 turns).

Pros:
- Fast and simple
- Lower token usage

Trade-off:
- Loses older facts unless reintroduced

### B) Summary + recent window (recommended)

Process:
1. Every N turns, summarize old messages into a short system note.
2. Clear old history.
3. Keep summary + most recent turns.

Example summary prompt:

```text
Summarize this conversation into:
1) user profile facts
2) stable preferences
3) open tasks/questions
Keep under 120 words.
```

Pros:
- Better long-session continuity
- Much lower token growth

Trade-off:
- Summaries can omit fine detail

### C) Retrieval-based memory (for advanced projects)

Store important facts as documents in a vector DB and retrieve relevant memories per user query.

Pros:
- Scales well for very long lifetimes
- More controllable than raw chat history

Trade-off:
- More moving parts (embedding model, retriever tuning)

**Persistence Layer**

### A) Persist chat history with Redis

Install:

```bash
pip install -U redis langchain-community
```

Run Redis locally (Docker):

```bash
docker run -d --name redis-chat -p 6379:6379 redis:7
```

Use `RedisChatMessageHistory` keyed by `session_id` so history survives app restarts.

Example pattern:

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379/0",
        key_prefix="day2:chat:"
    )
```


## Memory Optimization (Token Counting + Cost)
For Ollama, token counts are available in response metadata from ChatOllama.
```
# add near top
from dataclasses import dataclass, asdict
from datetime import datetime
import csv
import json

@dataclass
class TurnLog:
    ts: str
    session_id: str
    user_text: str
    bot_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    est_cost_usd: float

logs: list[TurnLog] = []
PRICE_PER_1K_INPUT = 0.0   # local ollama usually 0, set if you want cloud-equivalent estimate
PRICE_PER_1K_OUTPUT = 0.0
```
### Alternative Memory Strategy (Summary Compression)
ConversationSummaryMemory is older-style. With your current RunnableWithMessageHistory pattern, do summary compression manually every N turns.
```
SUMMARY_TRIGGER = 12  # messages in history

def maybe_summarize_history(chatbot, session_id: str) -> None:
    hist = get_session_history(session_id)
    if len(hist.messages) < SUMMARY_TRIGGER:
        return

    transcript = "\n".join([f"{m.type}: {m.content}" for m in hist.messages[:-4]])
    if not transcript.strip():
        return

    summary_resp = chatbot.invoke(
        {"input": f"Summarize this chat history in 6 bullet points:\n{transcript}"},
        config={"configurable": {"session_id": f"{session_id}-summarizer"}},
    )

    # keep recent context + one summary message
    from langchain_core.messages import SystemMessage
    recent = hist.messages[-4:]
    hist.clear()
    hist.add_message(SystemMessage(content=f"Conversation summary:\n{summary_resp.content}"))
    for m in recent:
        hist.add_message(m)
```

### B) Save structured logs for analytics

Log one record per turn with fields such as:
- timestamp
- session_id
- user_text
- bot_text
- prompt_tokens
- completion_tokens
- total_tokens
- latency_ms

### C) Export logs to JSON and CSV

JSON example:

```python
import json
with open("chat_logs.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
```

CSV example:

```python
import csv
with open("chat_logs.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
```

### D) Optional database storage

Start simple with SQLite for local labs, then move to PostgreSQL for multi-user production.

Suggested schema columns:
- `id`, `created_at`, `session_id`, `role`, `content`, `prompt_tokens`, `completion_tokens`

**Enhanced Features**

Goal: enrich each chat turn with small, fast metadata extraction while keeping latency low.

### A) Lightweight profile + sentiment pass per user turn

Recommended per-turn pipeline:
1. User sends message.
2. Run a small analysis pass (short prompt, deterministic output) to extract:
     - sentiment (`positive|neutral|negative`)
     - confidence (`0..1`)
     - profile candidates (name, interests, goals, style)
3. Validate and merge extracted fields into the user's profile.
4. Run the main assistant response using memory + updated profile context.
5. Log both answer and analysis metadata.

Why split analysis and response:
- Keeps profile updates predictable.
- Makes analytics easier.
- Lets you tune analysis prompt/model separately from chat prompt.

Minimal analysis schema:

```json
{
    "sentiment": "neutral",
    "confidence": 0.84,
    "name": null,
    "interests": [],
    "goals": [],
    "preferred_style": null
}
```

Classifier/extractor prompt example:

```text
Extract JSON only with keys:
sentiment (positive|neutral|negative), confidence (0..1),
name (string|null), interests (string[]), goals (string[]), preferred_style (string|null).

Rules:
- If unsure, return null or empty list.
- Do not invent facts.

Text: <user_input>
```

Merge logic (important):
1. Update `name` only if new value is non-empty and confidence >= 0.75.
2. Append `interests`/`goals` only when not already present (case-insensitive).
3. Update `preferred_style` only if user explicitly states preference.
4. Never delete existing stable facts unless user clearly corrects them.

Example helper function shape:

```python
def merge_profile(profile: dict, extracted: dict) -> dict:
        # apply confidence and dedup rules
        return profile
```

Recommended latency budget per turn:
- Analysis pass: <= 300 ms to 800 ms on local model.
- Main response: normal chat latency.

If latency is high:
1. Run analysis every 2-3 turns instead of every turn.
2. Use a smaller model for analysis.
3. Reduce extractor prompt length.

### B) Profile storage model

Store profile separately from raw message history.

Example profile document:

```json
{
    "user_id": "alice",
    "name": "Alice",
    "interests": ["ai", "ml"],
    "goals": ["learn langchain"],
    "preferred_style": "concise",
    "mood_trend": ["neutral", "positive"],
    "updated_at": "2026-03-05T10:15:00Z"
}
```

Suggested storage options:
1. Redis hash per user: fast read/write.
2. SQL table: easier long-term analytics and joins.
3. Hybrid: Redis (runtime) + SQL (durable archive).

### C) Multi-user support (CLI + API)

Core rule: every request must have a unique `session_id` (and ideally `user_id`).

Session id examples:
1. CLI lab: `session_id = input("username: ").strip().lower()`
2. API app: `session_id = auth_user_id` or `tenant:user` format.
3. Multi-device user: `session_id = f"{user_id}:{conversation_id}"`

Isolation requirements:
1. Use namespaced keys, for example:
     - history: `day2:chat:{session_id}`
     - profile: `day2:profile:{user_id}`
2. Never share one static session id like `user-1` in production.
3. Validate incoming `session_id` format to avoid key injection.

Access pattern per turn:
1. Resolve `user_id`, `session_id`.
2. Load profile for `user_id`.
3. Load chat history for `session_id`.
4. Run analysis pass, merge profile.
5. Generate assistant response with history + compact profile context.
6. Persist updated history, profile, and analytics row.

### D) End-of-session summary

When user types `exit` (or API conversation closes):
1. Generate a concise summary.
2. Save summary under session id.
3. Include key facts and unresolved tasks for next session continuity.

Suggested summary object:

```json
{
    "session_id": "alice:2026-03-05",
    "key_facts": ["name=Alice", "interest=ML"],
    "topics": ["LangChain memory", "Redis persistence"],
    "open_questions": ["How to evaluate token growth?"],
    "next_actions": ["Enable JSON log export"]
}
```

Summary prompt example:

```text
Create a session summary with sections:
1) Key user facts
2) Topics discussed
3) Open questions
4) Suggested next actions
Keep it concise and factual.
```

### Suggested build order

1. Add token logging and scenario comparison.
2. Add summary + recent window memory compression.
3. Add Redis-backed persistent message history.
4. Add JSON/CSV exports and basic analytics.
5. Add sentiment tracking and dynamic user profile.
6. Add multi-user session handling and end-of-session summaries.

---