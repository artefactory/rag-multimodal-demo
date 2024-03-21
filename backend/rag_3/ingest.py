"""Ingest PDF files into the vectorstore for RAG Option 3."""

import asyncio
import logging
import shutil
from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig
from tqdm.auto import tqdm
from unstructured.partition.pdf import partition_pdf

from backend.rag_3 import prompts
from backend.rag_3.config import validate_config
from backend.utils.elements import Image, Table, Text
from backend.utils.ingest import add_elements_to_multivector_retriever
from backend.utils.llm import get_text_llm, get_vision_llm
from backend.utils.retriever import get_retriever
from backend.utils.summarization import (
    generate_image_summaries,
    generate_text_summaries,
)
from backend.utils.unstructured import (
    load_chunking_func,
    select_images,
    select_tables,
    select_texts,
)

logger = logging.getLogger(__name__)


async def apply_summarize_text(text_list: list[Text], config: DictConfig) -> None:
    """Apply text summarization to a list of Text elements.

    The function directly modifies the Text elements inplace.

    Args:
        text_list (list[Text]): List of Text elements.
        config (DictConfig): Configuration object.
    """
    if config.ingest.summarize_text:
        str_list = [text.text for text in text_list]

        model = get_text_llm(config)

        text_summaries = await generate_text_summaries(
            str_list, prompt_template=prompts.TEXT_SUMMARIZATION_PROMPT, model=model
        )

        for text in text_list:
            text.set_summary(text_summaries.pop(0))

    else:
        logger.info("Skipping text summarization")

    return


async def apply_summarize_table(table_list: list[Table], config: DictConfig) -> None:
    """Apply table summarization to a list of Table elements.

    The function directly modifies the Table elements inplace.

    Args:
        table_list (list[Table]): List of Table elements.
        config (DictConfig): Configuration object.

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
                prompt_template=prompts.TABLE_SUMMARIZATION_PROMPT,
                model=model,
            )
        elif config.ingest.table_format == "image":
            img_base64_list = [table.base64 for table in table_list]
            img_mime_type_list = [table.mime_type for table in table_list]
            model = get_vision_llm(config)

            table_summaries = await generate_image_summaries(
                img_base64_list,
                img_mime_type_list,
                prompt=prompts.TABLE_SUMMARIZATION_PROMPT,
                model=model,
            )
        else:
            raise ValueError(f"Invalid table format: {table_format}")

        for table in table_list:
            table.set_summary(table_summaries.pop(0))

    else:
        logger.info("Skipping table summarization")

    return


async def apply_summarize_image(image_list: list[Image], config: DictConfig) -> None:
    """Apply image summarization to a list of Image elements.

    The function directly modifies the Image elements inplace.

    Args:
        image_list (list[Image]): List of Image elements.
        config (DictConfig): Configuration object.
    """
    img_base64_list = [image.base64 for image in image_list]
    img_mime_type_list = [image.mime_type for image in image_list]

    model = get_vision_llm(config)

    image_summaries = await generate_image_summaries(
        img_base64_list,
        img_mime_type_list,
        prompt=prompts.IMAGE_SUMMARIZATION_PROMPT,
        model=model,
    )

    for image in image_list:
        image.set_summary(image_summaries.pop(0))

    return


async def ingest_pdf(file_path: str | Path, config: DictConfig) -> None:
    """Ingest a PDF file.

    Args:
        file_path (str | Path): Path to the PDF file.
        config (DictConfig): Configuration object.
    """
    logger.info(f"Processing {file_path}")

    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        extract_image_block_types=["image", "table"],
        extract_image_block_to_payload=True,
    )

    # Get images
    images = select_images(
        raw_pdf_elements,
        metadata_keys=config.ingest.metadata_keys,
        min_size=config.ingest.image_min_size,
    )

    # Get chunks
    if config.ingest.chunking_enable:
        chunk_func = load_chunking_func(config)
        chunks = chunk_func(raw_pdf_elements)
    else:
        chunks = raw_pdf_elements

    # Get text, tables
    texts = select_texts(chunks, metadata_keys=config.ingest.metadata_keys)
    tables = select_tables(
        chunks,
        table_format=config.ingest.table_format,
        metadata_keys=config.ingest.metadata_keys,
        min_size=config.ingest.table_min_size,
    )

    # Summarize text
    await apply_summarize_text(texts, config)

    # Summarize tables
    await apply_summarize_table(tables, config)

    # Summarize images
    await apply_summarize_image(images, config)

    retriever = get_retriever(config)

    # Add texts to retriever
    logger.info("Adding texts to retriever")
    add_elements_to_multivector_retriever(
        elements=texts,
        retriever=retriever,
        vectorstore_source=config.ingest.vectorstore_source.text,
        docstore_source=config.ingest.docstore_source.text,
    )

    # Add tables to retriever
    logger.info("Adding tables to retriever")
    add_elements_to_multivector_retriever(
        elements=tables,
        retriever=retriever,
        vectorstore_source=config.ingest.vectorstore_source.table,
        docstore_source=config.ingest.docstore_source.table,
    )

    # Add images to retriever
    logger.info("Adding images to retriever")
    add_elements_to_multivector_retriever(
        elements=images,
        retriever=retriever,
        vectorstore_source=config.ingest.vectorstore_source.image,
        docstore_source=config.ingest.docstore_source.image,
    )

    # Export extracted elements
    if config.ingest.export_extracted:
        output_folder = Path(config.path.export_extracted) / Path(file_path).stem
        shutil.rmtree(output_folder, ignore_errors=True)
        output_folder.mkdir(parents=True, exist_ok=True)

        for idx, elem in enumerate(texts):
            elem.export(output_folder / "text", f"{idx:02d}")

        for idx, elem in enumerate(tables):
            elem.export(output_folder / "table", f"{idx:02d}")

        for idx, elem in enumerate(images):
            elem.export(output_folder / "image", f"{idx:02d}")

    return


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Ingest all PDF files in the docs folder.

    Args:
        config (DictConfig): Configuration object.
    """
    # Validate config
    _ = validate_config(config)

    # Clear database
    if config.ingest.clear_database:
        database_folder = Path(config.path.database)
        logger.info(f"Clearing database: {database_folder}")
        shutil.rmtree(database_folder, ignore_errors=True)

    docs_folder = Path(config.path.docs)

    for file_path in tqdm(sorted(docs_folder.glob("**/*.pdf"))):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(ingest_pdf(file_path, config))


if __name__ == "__main__":
    main()
