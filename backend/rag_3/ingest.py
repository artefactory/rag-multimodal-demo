"""Ingest PDF files into the vectorstore for RAG Option 3."""

import asyncio
import logging
import shutil
from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig
from tqdm.auto import tqdm

from backend.rag_3 import prompts
from backend.rag_3.config import validate_config
from backend.rag_components.ingest import (
    add_elements_to_multivector_retriever,
    apply_summarize_image,
    apply_summarize_table,
    apply_summarize_text,
)
from backend.rag_components.retriever import get_retriever
from backend.rag_components.unstructured import (
    load_chunking_func,
    load_partition_pdf_func,
    select_images,
    select_tables,
    select_texts,
)

logger = logging.getLogger(__name__)


async def ingest_pdf(file_path: str | Path, config: DictConfig) -> None:
    """Ingest a PDF file.

    Args:
        file_path (str | Path): Path to the PDF file.
        config (DictConfig): Configuration object.
    """
    logger.info(f"Processing {file_path}")

    # Get elements
    partition_pdf = load_partition_pdf_func(config)
    raw_pdf_elements = partition_pdf(filename=file_path)

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
    await apply_summarize_text(
        text_list=texts,
        config=config,
        prompt_template=prompts.TEXT_SUMMARIZATION_PROMPT,
    )

    # Summarize tables
    await apply_summarize_table(
        table_list=tables,
        config=config,
        prompt_template=prompts.TABLE_SUMMARIZATION_PROMPT,
    )

    # Summarize images
    await apply_summarize_image(
        image_list=images,
        config=config,
        prompt_template=prompts.IMAGE_SUMMARIZATION_PROMPT,
    )

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

    logger.info("Finished processing all PDF files")


if __name__ == "__main__":
    main()
