# pip install -U langchain langchain-ollama langchain-community faiss-cpu pypdf python-dotenv

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

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "RAG"
PDF_PATH = "islr.pdf"  # <-- change to your PDF filename

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index
emb = OllamaEmbeddings(model="nomic-embed-text")


# Ollama can fail when all PDF chunks are sent in one large request.
def build_faiss_index(documents, embedding_model, batch_size=32):
    texts = [document.page_content for document in documents]
    vectors = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(
            embedding_model.embed_documents(texts[start : start + batch_size])
        )

    return FAISS.from_embeddings(
        list(zip(texts, vectors)),
        embedding_model,
        metadatas=[document.metadata for document in documents],
    )


vs = build_faiss_index(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY from the provided context. If not found, say you don't know.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

# 5) Chain
llm = ChatOllama(model="gemma2:2b", temperature=0)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


parallel = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
