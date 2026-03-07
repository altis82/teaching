# Day 4 – Intelligent Agent with Tools and Reasoning (Advanced)

## 🎯 Learning Objectives

Build an AI agent that can **reason** about problems, **decide** which tools to use, and **execute** multi-step tasks autonomously using the ReAct pattern.

## 🧠 What You Will Learn

- **Tools**: Defining functions that agents can call
- **ReAct Pattern**: Reason → Act → Observe → Repeat
- **LangGraph Agent**: Production-ready agent using `create_react_agent`
- **Safe Execution**: How to let agents compute without security risks
- **Agent Reasoning**: Understanding how agents plan and execute strategies

## 🛠 Technical Requirements

### 1) Ollama Setup

```bash
ollama serve
ollama pull llama3.2:3b
```

### 2) Install Dependencies

```bash
cd /home/system/teaching/langchain/day4
pip install -r requirements.txt
```

> Note: This lab uses `langgraph` (installed with `langchain`). If missing:
> ```bash
> pip install langgraph
> ```

### 3) Project Structure

```
day4/
├── .env
├── config.py
├── main.py
├── requirements.txt
├── Day_4_Ollama_Tutorial.md
└── data/
    └── sales.csv          ← auto-generated on first run
```

---

## 🚀 Core Concepts: What is an Agent?

### Chain vs Agent

| | Chain | Agent |
|---|---|---|
| Flow | Fixed: A → B → C | Dynamic: decides at each step |
| Tools | Cannot call external functions | Can call tools based on need |
| Reasoning | None – just follows the pipeline | Thinks about what to do next |
| Use case | Predictable pipelines | Open-ended problem solving |

### The ReAct Pattern

ReAct = **Re**asoning + **Act**ing. The agent loops:

```
1. THINK  → "I need to calculate compound interest"
2. ACT    → calls calculator("10000 * (1 + 0.05) ** 10")
3. OBSERVE → gets "Result: 16288.95"
4. THINK  → "Now I have the answer, let me respond"
5. RESPOND → "After 10 years at 5%, $10,000 becomes $16,288.95"
```

If the first tool call doesn't give enough info, the agent loops back to step 1.

---

## 🔧 Step-by-Step Implementation

### Step 1: Define Tools

Tools are regular Python functions decorated with `@tool`. The docstring is **critical** – the agent reads it to decide when to use the tool.

#### Tool 1: Safe Calculator

```python
from langchain_core.tools import tool
import ast, operator

ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _safe_eval(node):
    """Evaluate an AST node using only allowed arithmetic operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_OPS:
            raise ValueError(f"Operator {op_type.__name__} not allowed")
        return ALLOWED_OPS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")

@tool
def calculator(expression: str) -> str:
    """Perform a mathematical calculation. Accepts expressions like
    '100000 * 0.05 * 5' or '(1 + 0.07) ** 10'."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"
```

**Why not `eval()`?**
- `eval()` can execute **any** Python code → security risk (e.g., `eval("__import__('os').system('rm -rf /')")`)
- `_safe_eval()` only allows arithmetic: `+ - * / **` → safe even with untrusted input.

#### Tool 2: CSV Analyzer

```python
import pandas as pd
from pathlib import Path

@tool
def analyze_csv(filename: str, query: str) -> str:
    """Analyze a CSV file with pandas. Provide the filename and a
    natural-language query (e.g. 'total revenue', 'month with highest sales')."""
    filepath = Path("data") / filename
    if not filepath.exists():
        available = [f.name for f in Path("data").glob("*.csv")]
        return f"File not found. Available: {available}"

    df = pd.read_csv(filepath)
    info = [
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Columns: {list(df.columns)}",
        f"First 5 rows:\n{df.head().to_string()}",
        f"Describe:\n{df.describe().to_string()}",
    ]
    return "\n".join(info) + f"\n\nUser query: {query}"
```

#### Tool 3: List Files

```python
@tool
def list_data_files() -> str:
    """List all CSV files available in the data/ folder."""
    files = list(Path("data").glob("*.csv"))
    if not files:
        return "No CSV files found."
    return "Available:\n" + "\n".join(f"  - {f.name}" for f in files)
```

**Key rule**: The agent **only knows about tools through their names and docstrings**. Write clear, specific docstrings.

---

### Step 2: Build the Agent

```python
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

def build_agent():
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        temperature=0.0,     # deterministic for tool use
    )

    tools = [calculator, analyze_csv, list_data_files]

    system_prompt = (
        "You are a helpful financial assistant with tools:\n"
        "1. calculator – for math\n"
        "2. analyze_csv – to inspect CSV data\n"
        "3. list_data_files – to see available files\n\n"
        "Always use calculator for math. Show reasoning step by step."
    )

    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent
```

**`create_react_agent`** from LangGraph:
- Wraps the LLM + tools into a ReAct loop.
- Handles the think → act → observe cycle automatically.
- Returns a graph that can be invoked like a regular chain.

**`temperature=0.0`**: For agents, deterministic output is important – you want reliable tool calls, not creative variations.

---

### Step 3: Invoke the Agent

```python
from langchain_core.messages import HumanMessage

agent = build_agent()

response = agent.invoke({
    "messages": [HumanMessage(content="What is 15% compound interest on $50,000 over 3 years?")]
})

# Print all messages to see the reasoning chain
for msg in response["messages"]:
    print(f"[{msg.type}] {msg.content[:200] if msg.content else ''}")
```

**Example output:**

```
[human] What is 15% compound interest on $50,000 over 3 years?
[ai] I need to calculate compound interest. The formula is: principal * (1 + rate)^years
[tool] Result: 76043.75
[ai] At 15% compound interest, $50,000 grows to $76,043.75 after 3 years.
     That's a gain of $26,043.75.
```

---

### Step 4: Interactive Loop

```python
def main():
    agent = build_agent()
    print("=== Financial Assistant Agent ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("you > ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

        for msg in response["messages"]:
            if msg.type == "ai" and msg.content:
                print(f"bot > {msg.content}")
            elif msg.type == "tool":
                print(f"  [tool: {msg.name}] {msg.content[:200]}")
        print()
```

Run:
```bash
python main.py
```

---

## 🧪 Test Scenarios

Try these questions to see the agent reason and use tools:

### Simple calculation:
```
you > If I invest $10,000 at 5% annual interest for 10 years, how much will I have?
```
Agent should: call `calculator("10000 * (1 + 0.05) ** 10")` → answer ~$16,288.95

### Multi-step reasoning:
```
you > What's the profit margin for each month in sales.csv?
```
Agent should: call `analyze_csv("sales.csv", "profit margin")` → see revenue & expenses → call `calculator` for each month or explain the formula.

### File discovery:
```
you > What data files do you have access to?
```
Agent should: call `list_data_files()` → list available CSVs.

### Complex scenario:
```
you > Look at sales.csv. Which month had the highest profit? What would that profit be worth in 5 years at 8% annual growth?
```
Agent should: analyze CSV → find best month → calculate future value.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| Agent doesn't call tools | Try a model with better tool-support: `ollama pull qwen3:8b` |
| `calculator` returns error | Check expression format – use `**` for power, not `^` |
| Agent loops forever | Some models struggle with ReAct. Add `recursion_limit=10` to `agent.invoke()` |
| CSV tool returns raw data | The agent should interpret it. If not, try rephrasing your question |

---

## ⭐ Advanced Extensions

### A) Add More Tools

```python
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Use requests to call a search API
    pass

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email notification."""
    # Implement email sending
    pass
```

### B) Agent Reasoning Logging

Log every step for debugging:

```python
response = agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    config={"recursion_limit": 15},
)

for i, msg in enumerate(response["messages"]):
    print(f"Step {i} [{msg.type}]: {msg.content[:100] if msg.content else '(tool call)'}")
```

### C) Human-in-the-Loop

Add approval for sensitive actions:

```python
@tool
def dangerous_operation(action: str) -> str:
    """Perform a sensitive operation. Requires user approval."""
    approval = input(f"⚠️  Agent wants to: {action}. Approve? (y/n): ")
    if approval.lower() != "y":
        return "Operation cancelled by user."
    # proceed with operation
    return "Operation completed."
```

### D) Tool Call Budget

Limit how many tools the agent can call per turn:

```python
response = agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    config={"recursion_limit": 5},   # max 5 tool calls
)
```

---

## 📝 Key Takeaways

1. **Agents = LLM + Tools + Reasoning loop** – the LLM decides what to do, not you.
2. **Tool docstrings are prompts** – write them as clearly as you'd write instructions for a person.
3. **Safety first** – never use `eval()` for agent-generated code. Use safe alternatives.
4. **Temperature = 0** for tool-calling agents – you want reliability, not creativity.
5. **ReAct pattern** is the foundation – think → act → observe → repeat.
6. **Start simple** – 2-3 well-defined tools > 10 vague tools.
