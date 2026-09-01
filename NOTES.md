# Learning notes

## Completed

### Lesson 1 — simple LLM call

The basic LangChain Expression Language flow is:

```text
PromptTemplate → ChatOllama → StrOutputParser
```

`ChatOllama(model="gemma2:2b")` runs the model locally through Ollama. No OpenAI API key is needed for this lesson.

### Lesson 2 — sequential chain

The chain runs two model calls in sequence:

```text
prompt1 → model → parser → {"text": ...} → prompt2 → model → parser
```

The `RunnableLambda` conversion is intentional. `StrOutputParser` returns a string, while `prompt2` expects a dictionary containing its `text` variable. Making that conversion explicit prevents the Zed/Pyright type error:

```python
def to_summary_input(text: str) -> dict[str, str]:
    return {"text": text}
```

## LangSmith run configuration

LangChain and LangGraph accept a `config` object when invoking a chain or workflow. This is useful for organizing and filtering runs in LangSmith:

```python
from langchain_core.runnables import RunnableConfig

run_config: RunnableConfig = {
    "run_name": "evaluate_upsc_essay",
    "tags": ["essay", "langgraph", "evaluation"],
    "metadata": {
        "model": "gemma2:2b",
        "dimensions": ["language", "analysis", "clarity"],
    },
}

result = workflow.invoke({"essay": essay}, config=run_config)
```

`run_name` gives the run a readable name. `tags` are searchable labels for grouping runs. `metadata` stores extra key-value information such as the model, document name, question, or configuration used for the run.

Use `RunnableConfig` as the type annotation when the editor reports that an inline config dictionary is incompatible with `invoke`. It makes the expected LangChain config shape explicit while preserving the tags and metadata in the LangSmith trace.

## Pending lessons

### Lesson 3 — RAG

Next, understand the full retrieval flow:

1. Load the PDF.
2. Split it into chunks.
3. Create local embeddings with `nomic-embed-text`.
4. Store and search vectors with FAISS.
5. Pass retrieved context to Gemma.
6. Compare the tracing and caching improvements across `v1`–`v4`.

The embedding model is separate from Gemma because chat models generate text, while embedding models convert text into vectors for similarity search.

### Lesson 4 — agents

Study how tools are defined, how the ReAct prompt chooses tools, and how `AgentExecutor` controls the loop. Verify Ollama tool-calling behavior before treating this example as production-ready.

### Lesson 5 — LangGraph

Study the state schema, parallel graph branches, reducers, and the final aggregation node. The workflow evaluates language, analysis, and clarity before calculating the average score.

## Next steps

- Run lessons 1 and 2 locally with Ollama.
- Enable LangSmith tracing and inspect the runs in the LangSmith dashboard.
- Complete and test the RAG lessons with questions grounded in `islr.pdf`.
- Record observations about latency, output quality, tracing, and caching here.
