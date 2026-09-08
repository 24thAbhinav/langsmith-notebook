# pip install -U langchain langchain-ollama langchain-community faiss-cpu pypdf python-dotenv langsmith

import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import traceable

load_dotenv()

PDF_PATH = "islr.pdf"  # <- change to your file


# ----------------- helpers (traced with tags & metadata) -----------------
@traceable(name="load_pdf", tags=["pdf", "loader"], metadata={"loader": "PyPDFLoader"})
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  # list[Document]


@traceable(
    name="split_documents",
    tags=["splitter"],
    metadata={"splitter": "RecursiveCharacterTextSplitter"},
)
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


@traceable(
    name="build_vectorstore",
    tags=["vectorstore"],
    metadata={"vectorstore": "FAISS"},
)
def build_vectorstore(splits):
    emb = OllamaEmbeddings(model="nomic-embed-text")
    texts = [document.page_content for document in splits]
    vectors = []
    for start in range(0, len(texts), 32):
        vectors.extend(emb.embed_documents(texts[start:start + 32]))

    return FAISS.from_embeddings(
        list(zip(texts, vectors)),
        emb,
        metadatas=[document.metadata for document in splits],
    )


# ----------------- parent setup function (traced) -----------------
@traceable(
    name="setup_pipeline",
    tags=["setup"],
    metadata={"chunk_size": 1000, "chunk_overlap": 150},
)
def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150):
    # ✅ These three steps are “clubbed” under this parent function
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    return vs


# ----------------- model, prompt, and run -----------------
llm = ChatOllama(model="gemma2:2b", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY from the provided context. If not found, say you don't know.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# ----------------- one top-level (root) run -----------------
@traceable(
    name="pdf_rag_full_run",
    tags=["rag", "full_run"],
    metadata={"model": "gemma2:2b", "embedding_model": "nomic-embed-text"},
)
def setup_pipeline_and_query(pdf_path: str, question: str):
    # Parent setup run (child of root)
    vectorstore = setup_pipeline(pdf_path, chunk_size=1000, chunk_overlap=150)

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    chain = parallel | prompt | llm | StrOutputParser()

    # This LangChain run stays under the same root (since we're inside this traced function)
    lc_config = {"run_name": "pdf_rag_query"}
    return chain.invoke(question, config=lc_config)


# ----------------- CLI -----------------
if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)
