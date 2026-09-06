# pip install -U langchain langchain-ollama langchain-community faiss-cpu pypdf python-dotenv langsmith

import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import traceable  # <-- key import

# --- LangSmith env (make sure these are set) ---
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=pdf_rag_demo

load_dotenv()

PDF_PATH = "islr.pdf"  # change to your file


# ---------- traced setup steps ----------
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
        vectors.extend(emb.embed_documents(texts[start : start + 32]))

    return FAISS.from_embeddings(
        list(zip(texts, vectors)),
        emb,
        metadatas=[document.metadata for document in splits],
    )


# You can also trace a “setup” umbrella span if you want:
@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs


# ---------- pipeline ----------
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


# Build the index under traced setup
vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run a query (also traced) ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

# Give the visible run name + tags/metadata so it’s easy to find:
config: RunnableConfig = {"run_name": "pdf_rag_query_v2"}

ans = chain.invoke(q, config=config)
print("\nA:", ans)
