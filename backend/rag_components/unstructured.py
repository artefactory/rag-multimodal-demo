"""Utility functions for working with unstructured elements."""

from collections.abc import Callable
from typing import Any

import unstructured.documents.elements as unstructured_elements
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from unstructured.documents.coordinates import RelativeCoordinateSystem

from .elements import Image, Table, TableImage, TableText, Text


def get_element_size(element: unstructured_elements.Element) -> tuple[float, float]:
    """Calculate the size of an element based on its coordinates.

    The coordinates are converted to a relative coordinate system (between 0 and 1).

    Args:
        element (unstructured_elements.Image): Unstructured element.

    Returns:
        tuple[float, float]: Width and height of the element.
    """
    coordinates = element.convert_coordinates_to_new_system(RelativeCoordinateSystem())

    min_x = min([c[0] for c in coordinates])
    max_x = max([c[0] for c in coordinates])
    min_y = min([c[1] for c in coordinates])
    max_y = max([c[1] for c in coordinates])

    width = max_x - min_x
    height = max_y - min_y

    return width, height


def select_images(
    raw_pdf_elements: list[unstructured_elements.Element],
    metadata_keys: list[str] | None = None,
    min_size: tuple[float, float] = (0, 0),
) -> list[Image]:
    """Extract images from a list of PDF elements and converts them to Image objects.

    Args:
        raw_pdf_elements (list[unstructured_elements.Element]): List of elements
            extracted from a PDF.
        metadata_keys (list[str], optional): List of metadata keys to extract for each
            image. Defaults to None.
        min_size (tuple[float, float], optional): Minimum relative size of the image in
            the format (width, height). Defaults to (0, 0).

    Returns:
        list[Image]: List of Image objects with the selected metadata.
    """
    images = []
    for element in raw_pdf_elements:
        if not isinstance(element, unstructured_elements.Image):
            continue

        width, height = get_element_size(element)
        if width < min_size[0] or height < min_size[1]:
            continue

        if (
            element.metadata.image_mime_type is None
            or element.metadata.image_base64 is None
        ):
            continue

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
        if not isinstance(element, unstructured_elements.CompositeElement):
            continue

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
    min_size: tuple[float, float] = (0, 0),
) -> list[Table]:
    """Extracts tables from a list of PDF elements and converts them into Table objects.

    Args:
        raw_pdf_elements (list[unstructured_elements.Element]): List of elements
            extracted from a PDF.
        table_format (str): Format to which the tables should be converted ('text',
            'html', or 'image').
        metadata_keys (list[str], optional): List of metadata keys to extract for each
            table. Defaults to None.
        min_size (tuple[float, float], optional): Minimum relative size of the table in
            the format (width, height). Defaults to (0, 0).

    Raises:
        ValueError: If the provided `table_format` is not supported.

    Returns:
        list[Table]: List of Table objects in the specified format with metadata.
    """
    tables = []
    for element in raw_pdf_elements:
        metadata = get_metadata(element, metadata_keys)
        if not isinstance(element, unstructured_elements.Table):
            continue

        width, height = get_element_size(element)
        if width < min_size[0] or height < min_size[1]:
            continue

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


def load_partition_pdf_func(config: DictConfig) -> Callable:
    """Load the partition_pdf function from the configuration.

    Args:
        config (DictConfig): Configuration object.

    Raises:
        ValueError: If the partition_pdf function is not callable.

    Returns:
        Callable: The partition_pdf function.
    """
    func = instantiate(config.ingest.partition_pdf_func)
    if not callable(func):
        raise ValueError("partition_pdf function must be callable")
    return func


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
