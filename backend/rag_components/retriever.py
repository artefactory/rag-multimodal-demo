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
    vectorstore_content: list[str],
    docstore_content: Sequence[str | Document],
    metadata_list: list[dict] | None = None,
    vectorstore_source: str = "summary",
    docstore_source: str = "content",
    id_key: str = "doc_id",
) -> None:
    """Add documents to the vectorstore and docstore of a MultiVectorRetriever.

    This function processes and adds document summaries to the retriever's vectorstore
    and the corresponding raw content to the retriever's docstore.
    The vectorstore is used to perform similarity searches, while the docstore holds
    the raw content of the documents that will be passed to the model in the RAG chain.

    Args:
        retriever (MultiVectorRetriever): Retriever to add the documents to.
        vectorstore_content (list[str]): List of documents to add to the
            vectorstore (usually the summaries).
        docstore_content (Sequence[str  |  Document]): List of documents to add to
            the docstore (usually the raw content).
        metadata_list (list[dict], optional): List of metadata dictionnaries associated
            with each document. Defaults to None.
        vectorstore_source (str, optional): Source name for the vectorstore. Defaults to
            "summary".
        docstore_source (str, optional): Source name for the docstore. Defaults to
            "content".
        id_key (str, optional): Key used for the unique document ID in the metadata.
            Defaults to "doc_id".

    Raises:
        ValueError: If the lengths of `vectorstore_content`, `docstore_content` and
            `metadata_list` do not match.
        ValueError: If the retriever is not an instance of MultiVectorRetriever.
    """
    if len(vectorstore_content) != len(docstore_content):
        raise ValueError(
            "The length of vectorstore_content and docstore_content must be the same"
        )
    if metadata_list is not None and len(vectorstore_content) != len(metadata_list):
        raise ValueError(
            "The length of vectorstore_content and metadata_list must be the same"
        )
    if len(vectorstore_content) == 0:
        return

    if not isinstance(retriever, MultiVectorRetriever):
        raise ValueError("retriever must be a MultiVectorRetriever")

    doc_ids = [str(uuid.uuid4()) for _ in vectorstore_content]

    if metadata_list is None:
        metadata_list = [{id_key: doc_ids[i]} for i in range(len(vectorstore_content))]
    else:
        metadata_list = [
            {id_key: doc_ids[i], **metadata_list[i]}
            for i in range(len(vectorstore_content))
        ]

    # Create Document objects from the summaries
    vectorstore_docs = [
        Document(
            page_content=s,
            metadata={**metadata_list[i], "source": vectorstore_source},
        )
        for i, s in enumerate(vectorstore_content)
    ]

    # Create Document objects from the raw content
    docstore_docs = [
        Document(
            page_content=c if isinstance(c, str) else c.page_content,
            metadata={**metadata_list[i], "source": docstore_source},
        )
        for i, c in enumerate(docstore_content)
    ]

    # Add summaries to the vectorstore and contents to the docstore
    retriever.vectorstore.add_documents(vectorstore_docs)
    retriever.docstore.mset(list(zip(doc_ids, docstore_docs, strict=False)))

    return
