"""Ingest utility functions."""

import logging
from collections.abc import Sequence

from langchain.retrievers.multi_vector import MultiVectorRetriever

from .elements import Element
from .retriever import add_documents_multivector

logger = logging.getLogger(__name__)


def get_attr_from_elements(elements: Sequence[Element], attr: str) -> list:
    """Get a specific attribute from a list of elements.

    Args:
        elements (list[Element]): List of elements.
        attr (str): Attribute to get from the elements.

    Raises:
        ValueError: If the attribute is not supported.

    Returns:
        list: List of the specified attribute from the elements.
    """
    match attr:
        case "content":
            return [element.get_content() for element in elements]
        case "summary":
            return [element.get_summary() for element in elements]
        case "metadata":
            return [element.get_metadata() for element in elements]
        case other:
            raise ValueError(f"Unsupported attribute: {other}")


def add_elements_to_multivector_retriever(
    elements: Sequence[Element],
    retriever: MultiVectorRetriever,
    vectorstore_source: str,
    docstore_source: str,
) -> None:
    """Add a list of elements to the multi-vector retriever.

    Args:
        elements (Sequence[Element]): List of elements to add.
        retriever (MultiVectorRetriever): Multi-vector retriever.
        vectorstore_source (str): Attribute of the elements to add to the vectorstore.
        docstore_source (str): Attribute of the elements to add to the docstore.
    """
    vectorstore_content = get_attr_from_elements(elements, vectorstore_source)
    docstore_content = get_attr_from_elements(elements, docstore_source)
    metadata_list = get_attr_from_elements(elements, "metadata")

    logging.info(f"Adding {vectorstore_source} to vectorstore.")
    logging.info(f"Adding {docstore_source} to docstore.")

    add_documents_multivector(
        retriever=retriever,
        vectorstore_content=vectorstore_content,
        docstore_content=docstore_content,
        metadata_list=metadata_list,
        vectorstore_source=vectorstore_source,
        docstore_source=docstore_source,
    )
