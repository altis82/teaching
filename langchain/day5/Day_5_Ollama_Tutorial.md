# Day 5 – Multi-Agent Orchestration with LangGraph (Advanced+)

> **This is production-level architecture.** You'll build a system where multiple specialized agents collaborate to produce a market analysis report.

## 🎯 Learning Objectives

Build a multi-agent system using **LangGraph** where each agent has a specialized role — research, analysis, writing, review — and they coordinate through a state graph.

## 🧠 What You Will Learn

- **Multi-Agent Architecture**: Agents with specialized roles
- **LangGraph StateGraph**: Production state machine for agent orchestration
- **Conditional Routing**: Dynamic branching based on quality scores
- **Self-Improvement Loop**: Critic agent sends work back for revision
- **State Management**: Passing data between agents through shared state

## 🛠 Technical Requirements

### 1) Ollama Setup

```bash
ollama serve
ollama pull llama3.2:3b
```

### 2) Install Dependencies

```bash
cd /home/system/teaching/langchain/day5
pip install -r requirements.txt
```

### 3) Project Structure

```
day5/
├── .env
├── config.py
├── main.py
├── requirements.txt
├── Day_5_Ollama_Tutorial.md
└── report_output.md        ← generated after running
```

---

## 🧩 Core Concept: LangGraph

### Why LangGraph?

| Approach | Limitation |
|---|---|
| Simple chain (Day 1-2) | Fixed flow, no branching |
| Agent + tools (Day 4) | Single agent, tools only |
| **LangGraph** | Multiple agents, branching, loops, checkpoints |

LangGraph models your workflow as a **directed graph**:
- **Nodes** = functions that process state
- **Edges** = transitions between nodes
- **Conditional edges** = dynamic routing based on state values
- **State** = shared data object passed through the graph

### Building Blocks

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. Define state shape
class MyState(TypedDict):
    input: str
    output: str

# 2. Define node functions (each takes and returns state)
def my_node(state: MyState) -> MyState:
    state["output"] = "processed: " + state["input"]
    return state

# 3. Build graph
graph = StateGraph(MyState)
graph.add_node("process", my_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)

# 4. Compile and run
app = graph.compile()
result = app.invoke({"input": "hello", "output": ""})
```

---

## 🚀 Step-by-Step: Multi-Agent Report Generator

### Architecture

```
[START]
   ↓
[Research Agent]  → gathers raw information
   ↓
[Analyst Agent]   → structures insights & metrics
   ↓
[Writer Agent]    → creates polished report draft
   ↓
[Critic Agent]    → reviews quality, gives score
   ↓
   ├── score < 0.8 AND revisions < 2 → back to [Writer Agent]
   └── score >= 0.8 OR max revisions → [Finalize] → [END]
```

Each agent is a **LangGraph node** — a function that reads/writes shared state.

---

### Step 1: Define Shared State

```python
from typing import TypedDict

class ReportState(TypedDict):
    topic: str               # user input
    research_notes: str      # output of research agent
    analysis: str            # output of analyst agent
    draft_report: str        # output of writer agent
    critique: str            # output of critic agent
    quality_score: float     # 0.0 to 1.0 from critic
    final_report: str        # finalized output
    revision_count: int      # how many times writer was called
```

**Why a shared state?**
- Every agent can read what previous agents produced.
- No need for complex message passing — just read/write fields.
- Easy to inspect and debug at any point.

---

### Step 2: Research Agent Node

```python
def research_node(state: ReportState) -> ReportState:
    topic = state["topic"]
    print(f"[Research Agent] Researching: {topic}")

    system = (
        "You are a research analyst. Gather comprehensive information: "
        "market size, key players, trends, growth drivers, challenges."
    )
    user = f"Research: {topic}"

    notes = run_prompt(system, user)
    state["research_notes"] = notes
    return state
```

**What it does:**
1. Reads `topic` from state.
2. Sends a research-focused prompt to the LLM.
3. Writes `research_notes` back to state.
4. Next agent (Analyst) will read these notes.

---

### Step 3: Analyst Agent Node

```python
def analyst_node(state: ReportState) -> ReportState:
    print("[Analyst Agent] Structuring insights...")

    system = (
        "You are a data analyst. Produce structured analysis:\n"
        "1. Executive summary\n"
        "2. Key statistics\n"
        "3. Trend analysis\n"
        "4. Competitive landscape\n"
        "5. Risks and opportunities"
    )
    user = f"Analyze:\n\n{state['research_notes']}"

    state["analysis"] = run_prompt(system, user)
    return state
```

**What it does:**
1. Reads `research_notes` from state.
2. Transforms unstructured notes → structured analysis.
3. Writes `analysis` to state.

---

### Step 4: Writer Agent Node

```python
def writer_node(state: ReportState) -> ReportState:
    print("[Writer Agent] Writing report...")

    revision_context = ""
    if state.get("critique"):
        revision_context = (
            f"\n\nPrevious feedback:\n{state['critique']}\n"
            "Address the feedback in this revision."
        )

    system = (
        "You are a professional report writer. Write a polished report with:\n"
        "- Title, Executive Summary, Market Overview\n"
        "- Key Players, Trends, Risks, Recommendations\n"
        "Use markdown formatting."
    )
    user = f"Topic: {state['topic']}\nAnalysis:\n{state['analysis']}{revision_context}"

    state["draft_report"] = run_prompt(system, user, temperature=0.7)
    state["revision_count"] = state.get("revision_count", 0) + 1
    return state
```

**Key detail:** If the Critic sent feedback (revision loop), the Writer includes it in the prompt. This is how the self-improvement loop works.

---

### Step 5: Critic Agent Node

```python
def critic_node(state: ReportState) -> ReportState:
    print("[Critic Agent] Reviewing...")

    system = (
        "Evaluate the report on: completeness, accuracy, clarity, "
        "structure, actionability.\n"
        "Respond in this format:\n"
        "SCORE: <0.0 to 1.0>\n"
        "FEEDBACK:\n<detailed feedback>"
    )
    user = f"Review:\n\n{state['draft_report']}"

    critique = run_prompt(system, user, temperature=0.2)
    state["critique"] = critique

    # Parse score
    score = 0.5
    for line in critique.split("\n"):
        if line.strip().upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
            break
    state["quality_score"] = score
    return state
```

**Why `temperature=0.2`?** The critic needs to be consistent and strict, not creative.

---

### Step 6: Routing Logic

```python
def route_after_critique(state: ReportState) -> str:
    if (state["quality_score"] < 0.8
            and state.get("revision_count", 0) < 2):
        return "revise"     # → back to writer
    return "finalize"       # → done
```

**This is the conditional edge.** It creates a feedback loop:
- Low quality + budget remaining → revise (writer gets critique feedback)
- Good quality OR max revisions reached → finalize

---

### Step 7: Build and Compile the Graph

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(ReportState)

# Add nodes
workflow.add_node("research", research_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)
workflow.add_node("finalize", finalize_node)

# Add edges (fixed transitions)
workflow.add_edge(START, "research")
workflow.add_edge("research", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", "critic")

# Conditional edge (dynamic routing)
workflow.add_conditional_edges("critic", route_after_critique, {
    "revise": "writer",
    "finalize": "finalize",
})
workflow.add_edge("finalize", END)

# Compile into runnable
app = workflow.compile()
```

**Visual:**
```
START → research → analyst → writer → critic
                               ↑         │
                               └─ revise ─┘ (if score < 0.8)
                                          │
                                     finalize → END (if score >= 0.8)
```

---

### Step 8: Run It

```bash
cd /home/system/teaching/langchain/day5
python main.py
```

```
=== Multi-Agent Report Generator ===

Enter report topic (or press Enter for default):
> The AI market in Vietnam

[1/4 Research Agent] Gathering information on: The AI market in Vietnam
  → 2340 chars of research notes

[2/4 Analyst Agent] Processing research into structured insights...
  → 1856 chars of analysis

[3/4 Writer Agent] Drafting the report...
  → 3200 chars (revision #1)

[4/4 Critic Agent] Reviewing the report...
  → Quality score: 0.72 (threshold: 0.80)
  → Score below 0.8, sending back for revision...

[3/4 Writer Agent] Drafting the report...
  → 3850 chars (revision #2)

[4/4 Critic Agent] Reviewing the report...
  → Quality score: 0.88

[Done] Finalizing report...

============================================================
FINAL REPORT
============================================================
# AI Market Analysis: Vietnam 2026
...
============================================================
Quality score: 0.88
Revisions: 2
Report saved to: report_output.md
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| Agent output too short | Use a larger model: `ollama pull qwen3:8b` |
| Score parsing fails | Check the Critic's raw output; adjust the prompt |
| Graph loops forever | `MAX_REVISIONS` in config.py limits revision count |
| Slow execution | Each agent is a separate LLM call. 4-6 calls total. Expect 1-3 min |
| `langgraph` not found | `pip install langgraph` |

---

## ⭐ Advanced Extensions

### A) Add Checkpointing (Resume Interrupted Runs)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "report-001"}}
result = app.invoke(initial_state, config=config)
```

### B) Parallel Agent Execution

If two agents don't depend on each other, run them in parallel:

```python
# Example: research agent + competitor-scan agent run simultaneously
workflow.add_edge(START, "research")
workflow.add_edge(START, "competitor_scan")
workflow.add_edge("research", "analyst")
workflow.add_edge("competitor_scan", "analyst")
```

### C) Add a FastAPI Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel

api = FastAPI()

class ReportRequest(BaseModel):
    topic: str

@api.post("/generate-report")
async def generate_report(req: ReportRequest):
    app = build_workflow()
    result = app.invoke({
        "topic": req.topic,
        "research_notes": "", "analysis": "",
        "draft_report": "", "critique": "",
        "quality_score": 0.0, "final_report": "",
        "revision_count": 0,
    })
    return {
        "report": result["final_report"],
        "quality_score": result["quality_score"],
        "revisions": result["revision_count"],
    }
```

### D) Self-Reflection Pattern

Let each agent evaluate its own work before passing to the next:

```python
def research_node_with_reflection(state):
    # First pass
    notes = run_prompt(research_system, user)

    # Self-reflection
    reflection = run_prompt(
        "Review your research notes. What's missing? What could be better?",
        notes
    )

    # Second pass with improvement
    improved = run_prompt(research_system, user + f"\nAlso address: {reflection}")
    state["research_notes"] = improved
    return state
```

### E) Monitoring & Logging

Log every node execution for analysis:

```python
import time

def timed_node(name, func):
    def wrapper(state):
        start = time.time()
        result = func(state)
        elapsed = time.time() - start
        print(f"  [{name}] completed in {elapsed:.1f}s")
        return result
    return wrapper

workflow.add_node("research", timed_node("research", research_node))
```

---

## 📝 Key Takeaways

1. **LangGraph = state machine for LLM apps** – nodes are agents, edges are transitions.
2. **Each agent has one job** – research, analyze, write, or review. Specialization > generalization.
3. **Shared state is the contract** – agents communicate through well-defined state fields.
4. **Conditional edges enable loops** – the critic can send work back for improvement.
5. **Always set a max revision limit** – otherwise a strict critic creates an infinite loop.
6. **This is production architecture** – add checkpointing, parallel execution, and API endpoints for real deployment.

---

## 📘 Curriculum Summary (Day 1-5)

| Day | Topic | Level | Key Concept |
|---|---|---|---|
| 1 | Hello LangChain | Beginner | PromptTemplate, Chain, OutputParser |
| 2 | Chatbot + Memory | Beginner→Intermediate | ChatHistory, RunnableWithMessageHistory |
| 3 | RAG System | Intermediate ⭐ | Embeddings, VectorStore, Retriever |
| 4 | Agent + Tools | Advanced | ReAct, Tool calling, LangGraph agent |
| 5 | Multi-Agent | Advanced+ | StateGraph, orchestration, self-improvement |
