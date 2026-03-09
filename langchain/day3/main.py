"""
Day 3 – RAG System: Question-Answering from Documents
Uses local Ollama for both LLM and embeddings.
"""

from pathlib import Path

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.output_parsers import StrOutputParser


from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL,
    TEMPERATURE, TOP_P, TOP_K,
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K, PDF_DIR,
)

VECTORSTORE_PATH = "vectorstore"

PDF_DIR="pdfs"
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
def load_documents(pdf_dir=PDF_DIR):
    #`DirectoryLoader will scan the directory for all pdf files
    # `
    loader = DirectoryLoader(
        path=pdf_dir, 
        glob="*.pdf", 
        # PyPDFLoader will extract text from each page --> document object
        # each document has `metadata` with source and page number
        loader_cls=PyPDFLoader,
        show_progress=True)
    docs =loader.load()
    print(f"Loaded {len(docs)} documents.")
    print(docs[0].page_content[:200])
    print(docs[0].metadata)
    return docs

def split_documents(docs):
    splitter =RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def get_embeddings():
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_EMBEDDING_MODEL
    )
def build_vectorstore(chunks):
    embeddings =get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=chunks, embedding=embeddings)
    vectorstore.save_local("vectorstore")
    return vectorstore

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."
         "If the content is insufficient, say so. Cite the source.\n\n"
         "Context: {context}"),
        ("human", "Question: {question}\n\nContext: {context}")
    ])
    llm=ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K
    )
    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'Unknown')},"
            f"page: {d.metadata.get('page', 'Unknown')}]:\n{d.page_content}"
            for d in docs
        )
    chain =(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def main():
    docs = load_documents(pdf_dir=PDF_DIR)
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    print(f"Vectorstore built with {len(vectorstore.docstore._dict)} documents.")
    chain, retriever= build_rag_chain(vectorstore)
    while True:

        user_input = input(f"You:").lower()
        user_input=user_input.strip()
        if user_input=="exit" or user_input=="quit":
            break

        answer = chain.invoke(user_input)
        print(answer)

if __name__ == "__main__":
    main()
