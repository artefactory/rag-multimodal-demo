"""This chain condenses the chat history and the question into a standalone question."""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel


class QuestionWithChatHistory(BaseModel):
    """Question with chat history."""

    question: str
    chat_history: str


class StandaloneQuestion(BaseModel):
    """Question rephrased to be standalone."""

    standalone_question: str


prompt = """\
Given the conversation history and the following question, can you rephrase the user's \
question in its original language so that it is self-sufficient. You are presented \
with a conversation that may contain some spelling mistakes and grammatical errors, \
but your goal is to understand the underlying question. Make sure to avoid the use of \
unclear pronouns.

If the question is already self-sufficient, return the original question. If it seem \
the user is authorizing the chatbot to answer without specific context, make sure to \
reflect that in the rephrased question.

Chat history: {chat_history}

Question: {question}
"""


def condense_question(llm: BaseChatModel) -> RunnableSequence:
    """Condense the chat history and the question into one standalone question.

    Args:
        llm (BaseChatModel): Language model used for generating the standalone question.

    Returns:
        RunnableSequence: Langchain sequence.
    """
    condense_question_prompt = PromptTemplate.from_template(
        prompt
    )  # chat_history, question

    standalone_question = condense_question_prompt | llm | StrOutputParser()

    typed_chain = standalone_question.with_types(
        input_type=QuestionWithChatHistory, output_type=StandaloneQuestion
    )
    return typed_chain
