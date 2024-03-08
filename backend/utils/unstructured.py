import unstructured.documents.elements as unstructured_elements
from hydra.utils import instantiate
from typing import Callable
from typing import Any
from backend.utils.elements import Image, Text, TableImage, TableText, Table


def select_images(
    raw_pdf_elements: list[unstructured_elements.Element],
    metadata_keys: list[str] = [],
) -> list[Image]:
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
    metadata_keys: list[str] = [],
) -> list[Text]:
    texts = []
    for element in raw_pdf_elements:
        if isinstance(element, unstructured_elements.CompositeElement):
            text = Text(
                text=element.text,
                metadata=get_metadata(element, metadata_keys),
            )
            texts.append(text)
    return texts


def select_tables(
    raw_pdf_elements: list[unstructured_elements.Element],
    table_format: str,
    metadata_keys: list[str] = [],
) -> list[Table]:
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
                    mime_type=element.metadata.mime_type,
                    format=table_format,
                    metadata=metadata,
                )
            else:
                raise ValueError(f"Invalid table format: {table_format}")
            tables.append(table)

    return tables


def load_chunking_func(config) -> Callable:
    return instantiate(config.ingest.chunking.func)


def get_metadata(
    elements: unstructured_elements.Element, keys: list[str]
) -> dict[str, Any]:
    return {key: getattr(elements.metadata, key) for key in keys}
