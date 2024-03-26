"""Ingest PDF files into the vectorstore for RAG Option 1."""

import logging
import shutil
from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig
from tqdm.auto import tqdm

from backend.rag_1.config import validate_config
from backend.rag_components.unstructured import (
    load_chunking_func,
    load_partition_pdf_func,
    select_images,
    select_tables,
    select_texts,
)
from backend.rag_components.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def ingest_pdf(file_path: str | Path, config: DictConfig) -> None:
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

    vectorstore = get_vectorstore(config)

    # Add texts to vectorstore
    logger.info("Adding texts to vectorstore")
    text_contents = [text.get_content() for text in texts]
    text_metadata = [text.get_metadata() for text in texts]

    vectorstore.add_texts(
        texts=text_contents,
        metadatas=text_metadata,
    )

    # Add tables to vectorstore
    logger.info("Adding tables to vectorstore")
    table_contents = [table.get_content() for table in tables]
    table_metadata = [table.get_metadata() for table in tables]

    vectorstore.add_texts(
        texts=table_contents,
        metadatas=table_metadata,
    )

    # Add images to retriever
    logger.info("Adding images to vectorstore")
    image_path = [image.get_local_path() for image in images]
    image_metadata = [image.get_metadata() for image in images]

    vectorstore.add_images(
        uris=image_path,
        metadatas=image_metadata,
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

    # Ingest all PDF files in the docs folder
    docs_folder = Path(config.path.docs)

    for file_path in tqdm(sorted(docs_folder.glob("**/*.pdf"))):
        ingest_pdf(file_path, config)

    logger.info("Finished processing all PDF files")


if __name__ == "__main__":
    main()
