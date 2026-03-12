from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import ast, operator
from pathlib import Path
import pandas as pd

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from config import *

# safe calculator using abstract syntax tree
ALLOWED_OPS={
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg

}

def _safe_eval(node):
    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op = ALLOWED_OPS.get(type(node.op))
        if op:
            return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op = ALLOWED_OPS.get(type(node.op))
        if op:
            return op(operand)
    raise ValueError(f"Unsupported operation: {type(node)}")

@tool 
def calculator(expression: str) -> str:
    """Perform a mathematical calculation. Accepts expressions like '100000 * 0.05 * 5' or '(1 + 0.07) ** 10'."""
    try:
        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")

@tool
def analyze_csv(file_name: str, query: str) -> str:
    """
    analyze csv file with pandas
    """
    file_path=Path("data") / file_name
    if not file_path.exists():
        raise ValueError(f"File not found: {file_name}")
    df = pd.read_csv(file_path)
    infor=[
        f"Shape: {df.shape[0]} rows, {df.shape[1]} columns",
        f"Columns: {', '.join(df.columns)}",
        f"First 5 rows:\n{df.head().to_string(index=False)}",
        f"Description:\n{df.describe().to_string()}"
    ]
    return "\n".join(infor)+ f"\n User query: {query}"

@tool
def list_data_files() -> str:
    """
    List all data files in the data directory.
    """
    data_dir = Path("data")
    if not data_dir.exists():
        raise ValueError("Data directory not found.")
    files = list(Path("data").glob("*.csv"))
    if not files:
        return "No CSV files found."
    return "\n".join([f.name for f in files])

def build_agent():
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.0
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
if __name__ == "__main__":
    main()