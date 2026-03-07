from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, RedisChatMessageHistory
from langchain_ollama import ChatOllama

from dataclasses import dataclass, asdict
from datetime import datetime
import csv
import json

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE, TOP_K, TOP_P,
    REDIS_URL,
    SUMMARIZE_THRESHOLD
)

# In-memory history store keyed by session id.
store: dict[str, BaseChatMessageHistory] = {}
# this is o monitoring the agent usage
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

log: list[TurnLog] = []
PRICE_PER_1K_INPUT=0.1
PRICE_PER_1K_OUTPUT =0.1
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

def get_session_history(session_id:str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix="chat_history"
        )
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

def maybe_summarize_history(chatbot, session_id: str) -> None:
    hist = get_session_history(session_id)
    if len(hist.messages) > SUMMARIZE_THRESHOLD:  # Example threshold, adjust as needed
        # Implement summarization logic here
        pass
    transcript ="\n".join([f"{msg.type}: {msg.content}" for msg in hist.messages[:-4]])
    if not transcript.strip():
        return
    summary_resp = chatbot.invoke(
        {"input": f"Please summarize the following conversation:\n{transcript}"},
        config={"configurable": {"session_id": f"{session_id}-summary"}}
    )
    from langchain_core.messages import SystemMessage
    recent = hist.messages[-4:]
    hist.clear()
    hist.add_message(SystemMessage(content=summary_resp.content))
    for m in recent:
        hist.add_message(m)

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
        maybe_summarize_history(chatbot, session_id)
        meta = getattr(response, "response_metadata", {})
        prompt_toks=int(meta.get("prompt_tokens", 0))
        completion_toks=int(meta.get("completion_tokens", 0))
        total_toks=prompt_toks+completion_toks
        est_cost_usd=(prompt_toks/1000)*PRICE_PER_1K_INPUT + (completion_toks/1000)*PRICE_PER_1K_OUTPUT

        log.append(TurnLog(
            ts=datetime.now().isoformat(),
            session_id=session_id,
            user_text=user_input,
            bot_text=response.content,
            prompt_tokens=prompt_toks,
            completion_tokens=completion_toks,
            total_tokens=total_toks,
            est_cost_usd=est_cost_usd
        ))

        print(f"bot > {response.content}\n")
        print(f"[tokens] Prompt: {prompt_toks}, Completion: {completion_toks}, Total: {total_toks}")
        print(f"[cost] Estimated cost: ${est_cost_usd:.4f}")


if __name__ == "__main__":
    main()
