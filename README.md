# LangSmith Masterclass

A small learning repository for experimenting with LangChain, LangSmith tracing, local Ollama models, RAG, agents, and LangGraph.

## Progress

- [x] Lesson 1 — simple LLM call
- [x] Lesson 2 — sequential chain
- [ ] Lesson 3 — RAG and LangSmith tracing
- [ ] Lesson 4 — tools and ReAct agents
- [ ] Lesson 5 — LangGraph workflows

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

## Notes

See [NOTES.md](NOTES.md) for learning notes and next steps.
