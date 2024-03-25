"""This chain fetches multimodal documents."""

from langchain.schema import format_document
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableSequence
from pydantic import BaseModel

from backend.utils.image import resize_base64_image

DOCUMENT_TEMPLATE = """\
Document metadata:
- Filename: {filename}
- Page number: {page_number}
Document content:
###
{page_content}
###
"""
DOCUMENT_PROMPT = PromptTemplate.from_template(template=DOCUMENT_TEMPLATE)


class Question(BaseModel):
    """Question to be answered."""

    question: str


class Documents(BaseModel):
    """Multimodal documents."""

    images: list[str]
    mime_types: list[str]
    texts: list[str]


def fetch_docs_chain(retriever: BaseRetriever) -> RunnableSequence:
    """Creates a chain that retrieves and processes multimodal documents.

    This chain first retrieves documents and then splits them into images and texts
    based on their metadata. It then formats the documents into a structure that
    separates base64-encoded images, their mime types, and text content.

    Args:
        retriever (BaseRetriever): Retriever that fetches documents.

    Returns:
        RunnableSequence: Langchain sequence.
    """
    relevant_documents = retriever | RunnableLambda(_split_image_text_types)
    typed_chain = relevant_documents.with_types(
        input_type=Question, output_type=Documents
    )
    return typed_chain


def _split_image_text_types(docs: list[Document]) -> dict[str, list]:
    """Split base64-encoded images and texts.

    Args:
        docs (list[Document]): List of documents.

    Returns:
        dict[str, list]: Dictionary containing lists of images, mime types, and texts.
    """
    img_base64_list = []
    img_mime_type_list = []
    text_list = []
    for doc in docs:
        match doc.metadata["type"]:
            case "text":
                formatted_doc = format_document(doc, DOCUMENT_PROMPT)
                text_list.append(formatted_doc)
            case "image":
                img = doc.page_content
                img = resize_base64_image(img)
                img_base64_list.append(resize_base64_image(img))
                img_mime_type_list.append(doc.metadata["mime_type"])
            case "table":
                if doc.metadata["format"] == "image":
                    img = doc.page_content
                    img = resize_base64_image(img)
                    img_base64_list.append(img)
                    img_mime_type_list.append(doc.metadata["mime_type"])
                else:
                    formatted_doc = format_document(doc, DOCUMENT_PROMPT)
                    text_list.append(formatted_doc)

    return {
        "images": img_base64_list,
        "mime_types": img_mime_type_list,
        "texts": text_list,
    }
