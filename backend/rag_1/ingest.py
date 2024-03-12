import shutil
from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig
from tqdm.auto import tqdm
from unstructured.partition.pdf import partition_pdf

from backend.rag_1.config import validate_config
from backend.utils.unstructured import (
    load_chunking_func,
    select_images,
    select_tables,
    select_texts,
)
from backend.utils.vectorstore import get_vectorstore


def ingest_pdf(file_path, config):
    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        extract_image_block_types=["image", "table"],
        extract_image_block_to_payload=True,
    )

    # Get images
    images = select_images(raw_pdf_elements, config.ingest.metadata_keys)

    # Get chunks
    if config.ingest.chunking_enable:
        chunk_func = load_chunking_func(config)
        chunks = chunk_func(raw_pdf_elements)
    else:
        chunks = raw_pdf_elements

    # Get text, tables
    texts = select_texts(chunks, config.ingest.metadata_keys)
    tables = select_tables(
        chunks, config.ingest.table_format, config.ingest.metadata_keys
    )

    vectorstore = get_vectorstore(config)

    # Add texts to retriever
    text_contents = [text.get_content() for text in texts]
    text_metadata = [text.get_metadata() for text in texts]

    vectorstore.add_texts(
        texts=text_contents,
        metadatas=text_metadata,
    )

    # Add tables to retriever
    table_contents = [table.get_content() for table in tables]
    table_metadata = [table.get_metadata() for table in tables]

    vectorstore.add_texts(
        texts=table_contents,
        metadatas=table_metadata,
    )

    # Add tables to retriever
    table_contents = [table.get_content() for table in tables]
    table_metadata = [table.get_metadata() for table in tables]

    vectorstore.add_texts(
        texts=table_contents,
        metadatas=table_metadata,
    )

    # Add images to retriever
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
def main(config: DictConfig):
    # Validate config
    _ = validate_config(config)

    docs_folder = Path(config.path.docs)

    for file_path in tqdm(sorted(docs_folder.glob("**/*.pdf"))):
        ingest_pdf(file_path, config)


if __name__ == "__main__":
    main()
