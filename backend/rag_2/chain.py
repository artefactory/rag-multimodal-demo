"""RAG chain for Option 2."""

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from omegaconf.dictconfig import DictConfig

from backend.utils.llm import get_text_llm
from backend.utils.retriever import get_retriever

from . import prompts


def get_chain(config: DictConfig) -> RunnableSequence:
    """Constructs a RAG pipeline that retrieves text data from documents.

    The pipeline consists of the following steps:
    1. Retrieval of documents using a retriever object.
    2. Prompting the model with the text data.
    4. Generating responses using a text language model.
    5. Parsing the string output.

    Args:
        config (DictConfig): Configuration object.

    Returns:
        RunnableSequence: RAG pipeline.
    """
    retriever = get_retriever(config)
    model = get_text_llm(config)

    # Prompt template
    prompt = ChatPromptTemplate.from_template(prompts.RAG_PROMPT)

    # Define the RAG pipeline
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain
