"""RAG pipeline with memory."""

from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.history import RunnableWithMessageHistory
from omegaconf import DictConfig
from pydantic import BaseModel

from backend.rag_components.chat_message_history import get_chat_message_history
from backend.rag_components.llm import get_text_llm

from .condense_question import condense_question


class QuestionWithHistory(BaseModel):
    """Question with chat history."""

    question: str
    chat_history: str


class Response(BaseModel):
    """Response to the question."""

    response: str


def construct_rag_with_history(
    base_chain: RunnableSequence,
    config: DictConfig,
) -> RunnableWithMessageHistory:
    """Constructs a RAG pipeline with memory.

    Args:
        base_chain (RunnableSequence): Base RAG pipeline.
        config (DictConfig): Configuration object.

    Returns:
        RunnableWithMessageHistory: RAG pipeline with memory.
    """
    text_llm = get_text_llm(config)

    reformulate_question = condense_question(text_llm)

    chain = reformulate_question | base_chain
    typed_chain = chain.with_types(input_type=QuestionWithHistory, output_type=Response)

    chain_with_mem = RunnableWithMessageHistory(
        typed_chain,
        lambda session_id: get_chat_message_history(config, session_id),
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_mem
