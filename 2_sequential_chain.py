from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_ollama import ChatOllama

load_dotenv()

import os

os.environ["LANGCHAIN_PROJECT"] = "2nd test"

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"],
)

model = ChatOllama(model="gemma2:2b")

parser = StrOutputParser()


def to_summary_input(text: str) -> dict[str, str]:
    return {"text": text}


config: RunnableConfig = {
    "run_name": "sequential_chain",
    "tags": ["sequential_chain", "report_generation", "summary_generation"],
    "metadata": {"model": "gemma2:2b", "parser": "StrOutputParser"},
}


chain = (
    prompt1
    | model
    | parser
    | RunnableLambda(to_summary_input)
    | prompt2
    | model
    | parser
)

result = chain.invoke({"topic": "Unemployment in India"}, config=config)

print(result)
