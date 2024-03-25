"""RAG chain for Option 2."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSequence,
    RunnableSerializable,
)
from omegaconf.dictconfig import DictConfig
from pydantic import BaseModel

from backend.rag_components.chain_links.rag_with_history import (
    construct_rag_with_history,
)
from backend.rag_components.chain_links.retrieve_and_format_text_docs import (
    fetch_docs_chain,
)
from backend.rag_components.llm import get_text_llm
from backend.rag_components.retriever import get_retriever

from . import prompts


class Question(BaseModel):
    """Question to be answered."""

    question: str


class Response(BaseModel):
    """Response to the question."""

    response: str


def get_base_chain(config: DictConfig) -> RunnableSequence:
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
            "context": fetch_docs_chain(retriever),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )
    typed_chain = chain.with_types(input_type=str, output_type=Response)

    return typed_chain


def get_chain(config: DictConfig) -> RunnableSerializable:
    """Get the appropriate RAG pipeline based on the configuration.

    Args:
        config (DictConfig): Configuration object.

    Returns:
        RunnableSerializable: RAG pipeline.
    """
    base_chain = get_base_chain(config)
    if config.rag.enable_chat_memory:
        chain_with_mem = construct_rag_with_history(base_chain, config)
        return chain_with_mem
    return base_chain
