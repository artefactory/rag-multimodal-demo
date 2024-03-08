from hydra.utils import instantiate
from langchain_core.retrievers import BaseRetriever

import uuid

from langchain_core.documents import Document

from typing import Optional


def get_retriever(config) -> BaseRetriever:
    return instantiate(config.retriever)


# Helper function to add documents to the vectorstore and docstore
def add_documents(
    retriever,
    doc_summaries: list[str],
    doc_contents: Optional[list[Document]] = None,
    doc_contents_str: Optional[list[str]] = None,
    doc_metadata: Optional[list[dict]] = None,
    id_key: str = "doc_id",
) -> None:
    if doc_contents is None and doc_contents_str is None:
        raise ValueError("Either doc_contents or doc_contents_str must be provided")

    if doc_contents is not None and doc_contents_str is not None:
        raise ValueError(
            "Only one of doc_contents or doc_contents_str must be provided"
        )

    doc_ids = [str(uuid.uuid4()) for _ in doc_summaries]

    # If doc_contents_str is provided, create Document objects from the strings
    if doc_contents_str is not None:
        doc_contents = [
            Document(
                page_content=s,
                metadata={
                    id_key: doc_ids[i],
                    **(doc_metadata[i] if doc_metadata else {}),
                },
            )
            for i, s in enumerate(doc_contents_str)
        ]

    assert len(doc_summaries) == len(doc_contents) == len(doc_ids)

    # Create Document objects from the summaries
    summary_docs = [
        Document(
            page_content=s,
            metadata={
                id_key: doc_ids[i],
                **(doc_metadata[i] if doc_metadata else {}),
            },
        )
        for i, s in enumerate(doc_summaries)
    ]

    # Add documents to the vectorstore and docstore
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    return
