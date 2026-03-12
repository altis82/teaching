"""
Day 4 – Intelligent Agent with Tools and Reasoning
Uses local Ollama with the ReAct agent pattern.
"""

import ast
import operator
import os
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE, TOP_P, TOP_K, CSV_DIR


# ── Tool 1: Safe Calculator ─────────────────────────────────────────
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
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return ALLOWED_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Perform a mathematical calculation. Accepts expressions like '100000 * 0.05 * 5' or '(1 + 0.07) ** 10'."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}. Please provide a valid arithmetic expression."


# ── Tool 2: CSV Data Analyzer ───────────────────────────────────────
@tool
def analyze_csv(filename: str, query: str) -> str:
    """Analyze a CSV file with pandas. Provide the filename (in the data/ folder) and a natural-language query describing what to compute (e.g., 'total revenue', 'average salary', 'month with highest sales')."""
    filepath = Path(CSV_DIR) / filename
    if not filepath.exists():
        available = [f.name for f in Path(CSV_DIR).glob("*.csv")]
        return f"File '{filename}' not found. Available: {available}"

    try:
        df = pd.read_csv(filepath)
        info_lines = [
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns: {list(df.columns)}",
            f"Dtypes:\n{df.dtypes.to_string()}",
            f"\nFirst 5 rows:\n{df.head().to_string()}",
            f"\nDescribe:\n{df.describe().to_string()}",
        ]
        return "\n".join(info_lines) + f"\n\nUser query: {query}"
    except Exception as e:
        return f"Error reading CSV: {e}"


# ── Tool 3: List Available Files ─────────────────────────────────────
@tool
def list_data_files() -> str:
    """List all CSV files available in the data/ folder."""
    data_path = Path(CSV_DIR)
    if not data_path.exists():
        return f"Data directory '{CSV_DIR}/' does not exist."
    files = list(data_path.glob("*.csv"))
    if not files:
        return f"No CSV files found in '{CSV_DIR}/'."
    return "Available CSV files:\n" + "\n".join(f"  - {f.name}" for f in files)


# ── Build Agent ──────────────────────────────────────────────────────
def build_agent():
    """Create a ReAct agent with the defined tools."""
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    tools = [calculator, analyze_csv, list_data_files]

    system_prompt = (
        "You are a helpful financial assistant. You have access to tools:\n"
        "1. calculator – for math expressions\n"
        "2. analyze_csv – to inspect and analyze CSV data files\n"
        "3. list_data_files – to see available CSV files\n\n"
        "Always use the calculator for math instead of computing in your head.\n"
        "Show your reasoning step by step."
    )

    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent


# ── Interactive Loop ─────────────────────────────────────────────────
def main():
    # Create sample CSV if data/ is empty
    data_dir = Path(CSV_DIR)
    data_dir.mkdir(exist_ok=True)
    sample_csv = data_dir / "sales.csv"
    if not sample_csv.exists():
        df = pd.DataFrame({
            "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            "revenue": [12000, 15000, 13500, 17000, 16500, 19000,
                        21000, 20000, 18500, 22000, 24000, 26000],
            "expenses": [8000, 9000, 8500, 10000, 9500, 11000,
                         12000, 11500, 10500, 13000, 14000, 15000],
            "customers": [120, 145, 130, 170, 160, 190,
                          210, 200, 185, 220, 240, 260],
        })
        df.to_csv(sample_csv, index=False)
        print(f"[setup] Created sample data: {sample_csv}")

    agent = build_agent()

    print("\n=== Financial Assistant Agent ===")
    print("Ask financial questions. The agent can calculate and analyze CSV data.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("you > ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("bye")
            break

        print()
        response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

        # Print the final AI message
        for msg in response["messages"]:
            if msg.type == "ai" and msg.content:
                print(f"bot > {msg.content}")
            elif msg.type == "tool":
                print(f"  [tool: {msg.name}] {msg.content[:200]}")
        print()


if __name__ == "__main__":
    main()
