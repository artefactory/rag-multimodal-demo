"""Ingest utility functions."""

import logging
from collections.abc import Sequence

from langchain.retrievers.multi_vector import MultiVectorRetriever
from omegaconf.dictconfig import DictConfig

from .elements import Element, Image, Table, Text
from .llm import get_text_llm, get_vision_llm
from .retriever import add_documents_multivector
from .summarization import (
    generate_image_summaries,
    generate_text_summaries,
)

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


async def apply_summarize_text(
    text_list: list[Text],
    config: DictConfig,
    prompt_template: str,
    chain_config: dict | None = None,
) -> None:
    """Apply text summarization to a list of Text elements.

    The function directly modifies the Text elements inplace.

    Args:
        text_list (list[Text]): List of Text elements.
        config (DictConfig): Configuration object.
        prompt_template (str): Prompt template for the summarization.
        chain_config (dict, optional): Configuration for the chain. Defaults to None.
    """
    if config.ingest.summarize_text:
        str_list = [text.text for text in text_list]

        model = get_text_llm(config)

        text_summaries = await generate_text_summaries(
            str_list,
            prompt_template=prompt_template,
            model=model,
            chain_config=chain_config,
        )

        for text in text_list:
            text.set_summary(text_summaries.pop(0))

    else:
        logger.info("Skipping text summarization")

    return


async def apply_summarize_table(
    table_list: list[Table],
    config: DictConfig,
    prompt_template: str,
    chain_config: dict | None = None,
) -> None:
    """Apply table summarization to a list of Table elements.

    The function directly modifies the Table elements inplace.

    Args:
        table_list (list[Table]): List of Table elements.
        config (DictConfig): Configuration object.
        prompt_template (str): Prompt template for the summarization.
        chain_config (dict, optional): Configuration for the chain. Defaults to None.

    Raises:
        ValueError: If the table format is "image" and summarize_table is False.
        ValueError: If the table format is invalid.
    """
    if config.ingest.summarize_table:
        table_format = config.ingest.table_format
        if table_format in ["text", "html"]:
            str_list = [table.text for table in table_list]

            model = get_text_llm(config)

            table_summaries = await generate_text_summaries(
                str_list,
                prompt_template=prompt_template,
                model=model,
                chain_config=chain_config,
            )
        elif config.ingest.table_format == "image":
            img_base64_list = [table.base64 for table in table_list]
            img_mime_type_list = [table.mime_type for table in table_list]
            model = get_vision_llm(config)

            table_summaries = await generate_image_summaries(
                img_base64_list,
                img_mime_type_list,
                prompt=prompt_template,
                model=model,
                chain_config=chain_config,
            )
        else:
            raise ValueError(f"Invalid table format: {table_format}")

        for table in table_list:
            table.set_summary(table_summaries.pop(0))

    else:
        logger.info("Skipping table summarization")

    return


async def apply_summarize_image(
    image_list: list[Image],
    config: DictConfig,
    prompt_template: str,
    chain_config: dict | None = None,
) -> None:
    """Apply image summarization to a list of Image elements.

    The function directly modifies the Image elements inplace.

    Args:
        image_list (list[Image]): List of Image elements.
        config (DictConfig): Configuration object.
        prompt_template (str): Prompt template for the summarization.
        chain_config (dict, optional): Configuration for the chain. Defaults to None.
    """
    img_base64_list = [image.base64 for image in image_list]
    img_mime_type_list = [image.mime_type for image in image_list]

    model = get_vision_llm(config)

    image_summaries = await generate_image_summaries(
        img_base64_list,
        img_mime_type_list,
        prompt=prompt_template,
        model=model,
        chain_config=chain_config,
    )

    for image in image_list:
        image.set_summary(image_summaries.pop(0))

    return
