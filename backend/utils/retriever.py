"""Utility functions for document retrievers."""

import uuid
from collections.abc import Sequence

from hydra.utils import instantiate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from omegaconf.dictconfig import DictConfig


def get_retriever(config: DictConfig) -> BaseRetriever:
    """Instantiate and return a BaseRetriever object based on the provided config.

    Args:
        config (DictConfig): Configuration object.

    Raises:
        ValueError: If the instantiated object is not a BaseRetriever.

    Returns:
        BaseRetriever: Instance of BaseRetriever.
    """
    object = instantiate(config.retriever)
    if not isinstance(object, BaseRetriever):
        raise ValueError(f"Expected a BaseRetriever object, got {type(object)}")
    return object


def add_documents_multivector(
    retriever: MultiVectorRetriever,
    summary_list: list[str],
    content_list: Sequence[str | Document],
    metadata_list: list[dict] | None = None,
    id_key: str = "doc_id",
) -> None:
    """Add documents to the vectorstore and docstore of a MultiVectorRetriever.

    This function processes and adds document summaries to the retriever's vectorstore
    and the corresponding raw content to the retriever's docstore.
    The vectorstore is used to perform similarity searches, while the docstore holds
    the raw content of the documents that will be passed to the model in the RAG chain.

    Args:
        retriever (MultiVectorRetriever): Retriever to add the documents to.
        summary_list (list[str]): List of document summaries.
        content_list (Sequence[str  |  Document]): List of document raw contents.
        metadata_list (list[dict], optional): List of metadata dictionnaries associated
            with each document. Defaults to None.
        id_key (str, optional): Key used for the unique document ID in the metadata.
            Defaults to "doc_id".

    Raises:
        ValueError: If the lengths of `summary_list`, `content_list` and `metadata_list`
            do not match.
        ValueError: If the retriever is not an instance of MultiVectorRetriever.
    """
    if len(summary_list) != len(content_list):
        raise ValueError("The length of summary_list and content_list must be the same")
    if metadata_list is not None and len(summary_list) != len(metadata_list):
        raise ValueError(
            "The length of summary_list and metadata_list must be the same"
        )

    if not isinstance(retriever, MultiVectorRetriever):
        raise ValueError("retriever must be a MultiVectorRetriever")

    doc_ids = [str(uuid.uuid4()) for _ in summary_list]

    if metadata_list is None:
        metadata_list = [{id_key: doc_ids[i]} for i in range(len(summary_list))]
    else:
        metadata_list = [
            {id_key: doc_ids[i], **metadata_list[i]} for i in range(len(summary_list))
        ]

    # Create Document objects from the summaries
    summary_docs = [
        Document(
            page_content=s,
            metadata=metadata_list[i],
        )
        for i, s in enumerate(summary_list)
    ]

    # Create Document objects from the raw content
    content_docs = [
        Document(
            page_content=c if isinstance(c, str) else c.page_content,
            metadata=metadata_list[i],
        )
        for i, c in enumerate(content_list)
    ]

    # Add summaries to the vectorstore and contents to the docstore
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, content_docs, strict=False)))

    return
