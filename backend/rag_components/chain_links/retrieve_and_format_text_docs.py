"""This chain fetches text documents and combines them into a single string."""

from langchain.schema import format_document
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableSequence
from pydantic import BaseModel

DOCUMENT_TEMPLATE = """\
Document metadata:
- Filename: {filename}
- Page number: {page_number}
Document content:
###
{page_content}
###
"""


class Question(BaseModel):
    """Question to be answered."""

    question: str


class Documents(BaseModel):
    """Text documents."""

    documents: str


def fetch_docs_chain(retriever: BaseRetriever) -> RunnableSequence:
    """Creates a chain that retrieves and formats text documents.

    This chain uses the provided retriever to fetch text documents and then combines
    them into a single string formatted according to a predefined template. The
    resulting string includes metadata and content for each document.

    Args:
        retriever (BaseRetriever): Retriever that fetches documents.

    Returns:
        RunnableSequence: Langchain sequence.
    """
    relevant_documents = retriever | RunnableLambda(_combine_documents)
    typed_chain = relevant_documents.with_types(
        input_type=Question, output_type=Documents
    )
    return typed_chain


def _combine_documents(docs: list[Document], document_separator: str = "\n\n") -> str:
    r"""Combine a list of text documents into a single string.

    Args:
        docs (list[Document]): List of documents.
        document_separator (str, optional): String to insert between each formatted
            document. Defaults to "\n\n".

    Returns:
        str: Single string containing all formatted documents
    """
    document_prompt = PromptTemplate.from_template(template=DOCUMENT_TEMPLATE)
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)
