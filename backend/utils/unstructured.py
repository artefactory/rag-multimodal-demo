"""Utility functions for working with unstructured elements."""

from collections.abc import Callable
from typing import Any

import unstructured.documents.elements as unstructured_elements
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from backend.utils.elements import Image, Table, TableImage, TableText, Text


def select_images(
    raw_pdf_elements: list[unstructured_elements.Element],
    metadata_keys: list[str] | None = None,
) -> list[Image]:
    """Extract images from a list of PDF elements and converts them to Image objects.

    Args:
        raw_pdf_elements (list[unstructured_elements.Element]): List of elements
            extracted from a PDF.
        metadata_keys (list[str], optional): List of metadata keys to extract for each
            image. Defaults to None.

    Returns:
        list[Image]: List of Image objects with the selected metadata.
    """
    images = []
    for element in raw_pdf_elements:
        if isinstance(element, unstructured_elements.Image):
            image = Image(
                base64=element.metadata.image_base64,
                mime_type=element.metadata.image_mime_type,
                metadata=get_metadata(element, metadata_keys),
            )
            images.append(image)
    return images


def select_texts(
    raw_pdf_elements: list[unstructured_elements.Element],
    metadata_keys: list[str] | None = None,
) -> list[Text]:
    """Extract texts from a list of PDF elements and converts them to Text objects.

    Args:
        raw_pdf_elements (list[unstructured_elements.Element]): List of elements
            extracted from a PDF.
        metadata_keys (list[str], optional): List of metadata keys to extract for each
            image. Defaults to None.

    Returns:
        list[Text]: List of Text objects with the selected metadata.
    """
    texts = []
    for element in raw_pdf_elements:
        if isinstance(element, unstructured_elements.CompositeElement):
            text = Text(
                text=element.text,
                format="text",
                metadata=get_metadata(element, metadata_keys),
            )
            texts.append(text)
    return texts


def select_tables(
    raw_pdf_elements: list[unstructured_elements.Element],
    table_format: str,
    metadata_keys: list[str] | None = None,
) -> list[Table]:
    """Extracts tables from a list of PDF elements and converts them into Table objects.

    Args:
        raw_pdf_elements (list[unstructured_elements.Element]): List of elements
            extracted from a PDF.
        table_format (str): Format to which the tables should be converted ('text',
            'html', or 'image').
        metadata_keys (list[str], optional): List of metadata keys to extract for each
            table. Defaults to None.

    Raises:
        ValueError: If the provided `table_format` is not supported.

    Returns:
        list[Table]: List of Table objects in the specified format with metadata.
    """
    tables = []
    for element in raw_pdf_elements:
        metadata = get_metadata(element, metadata_keys)
        if isinstance(element, unstructured_elements.Table):
            if table_format == "text":
                table = TableText(
                    text=element.text,
                    format=table_format,
                    metadata=metadata,
                )
            elif table_format == "html":
                table = TableText(
                    text=element.metadata.text_as_html,
                    format=table_format,
                    metadata=metadata,
                )
            elif table_format == "image":
                table = TableImage(
                    base64=element.metadata.image_base64,
                    mime_type=element.metadata.image_mime_type,
                    format=table_format,
                    metadata=metadata,
                )
            else:
                raise ValueError(f"Invalid table format: {table_format}")
            tables.append(table)

    return tables


def load_chunking_func(config: DictConfig) -> Callable:
    """Load the chunking function from the configuration.

    Args:
        config (DictConfig): Configuration object.

    Raises:
        ValueError: If the chunking function is not callable.

    Returns:
        Callable: The chunking function.
    """
    func = instantiate(config.ingest.chunking_func)
    if not callable(func):
        raise ValueError("Chunking function must be callable")
    return func


def get_metadata(
    elements: unstructured_elements.Element, keys: list[str] | None = None
) -> dict[str, Any]:
    """Get the metadata of an Unstructured Element.

    Args:
        elements (unstructured_elements.Element): Unstructured Element object.
        keys (list[str], optional): List of metadata keys to include in the output.
            Defaults to None.

    Returns:
        dict[str, Any]: Dictionary containing the metadata of the element.
    """
    if keys is None:
        keys = []
    return {key: getattr(elements.metadata, key) for key in keys}
