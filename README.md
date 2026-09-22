# LangSmith Masterclass

A small learning repository for experimenting with LangChain, LangSmith tracing, local Ollama models, RAG, agents, and LangGraph.

## Progress

- [x] Lesson 1 — simple LLM call
- [x] Lesson 2 — sequential chain
- [x] Lesson 3 — RAG and LangSmith tracing (`v1`–`v4`)
- [x] Lesson 4 — tools and ReAct agents
- [x] Lesson 5 — LangGraph workflows

All five lessons are implemented and traced end to end. See [NOTES.md](NOTES.md) for detailed trace walkthroughs, latency comparisons, and screenshots.

The code currently uses Ollama for local inference:

- Chat model: `gemma2:2b`
- Embedding model for RAG: `nomic-embed-text`

## Setup

Install and start [Ollama](https://ollama.com), then download the models:

```bash
ollama pull gemma2:2b
ollama pull nomic-embed-text
```

Create or activate the project environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

In Zed, select `venv` as the Python toolchain using `toolchain: select`.

## Environment variables

Copy `.env.example` to `.env` if you need LangSmith tracing:

```bash
cp .env.example .env
```

Then add your LangSmith values:

```dotenv
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=langsmith-masterclass
```

`.env` is ignored by git and should never be committed.

Two environment-variable families work with the installed `langsmith`/`langchain-core`:

- Legacy: `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`
- Current: `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`

Individual lessons override the project at runtime with `os.environ` so their traces land in separate dashboard projects (`2nd test`, `RAG`, `ReAct Agent`, `LANGGRAPH`). `3_rag_v1.py` uses the `LANGSMITH_PROJECT` name while the others use `LANGCHAIN_PROJECT` — both are honored by LangSmith.

## Run lessons

Run commands from this directory:

```bash
venv/bin/python 1_simple_llm_call.py
venv/bin/python 2_sequential_chain.py
venv/bin/python 3_rag_v1.py
venv/bin/python 3_rag_v2.py
venv/bin/python 3_rag_v3.py
venv/bin/python 3_rag_v4.py
venv/bin/python 4_agent.py
venv/bin/python 5_langgraph.py
```

The RAG examples use `islr.pdf` and prompt for a question in the terminal. The later examples also demonstrate LangSmith tracing, tools, and graph-based workflows.

`3_rag_v4.py` caches its FAISS index under `.indices/<hash>/` (gitignored). The hash is derived from the PDF contents, chunk size, chunk overlap, and embedding model, so a cache hit skips PDF parsing and embedding generation on repeat runs.

## Project map

| File | Topic |
| --- | --- |
| `1_simple_llm_call.py` | Prompt → Ollama model → output parser |
| `2_sequential_chain.py` | Two prompts composed with LCEL |
| `3_rag_v1.py` | Basic PDF RAG with FAISS |
| `3_rag_v2.py` | RAG with traced setup and query runs |
| `3_rag_v3.py` | Nested LangSmith tracing |
| `3_rag_v4.py` | Cached local FAISS indexes |
| `4_agent.py` | ReAct agent with search and weather tools |
| `5_langgraph.py` | Parallel essay evaluation workflow |
| `assets/` | LangSmith trace screenshots referenced by NOTES.md |
| `.indices/` | Gitignored FAISS cache written by `3_rag_v4.py` |

### LangSmith projects used

| Lesson | `os.environ` project |
| --- | --- |
| 1 | from `.env` (`langsmith-masterclass`) |
| 2 | `2nd test` |
| 3 (`v1`) | `RAG` (via `LANGSMITH_PROJECT`) |
| 3 (`v2`–`v4`) | from `.env`; `v2` comment suggests `pdf_rag_demo` |
| 4 | `ReAct Agent` |
| 5 | `LANGGRAPH` |

## Notes

See [NOTES.md](NOTES.md) for learning notes, trace walkthroughs, and screenshots.
