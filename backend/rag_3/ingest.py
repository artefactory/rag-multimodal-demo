import asyncio
import shutil
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from unstructured.partition.pdf import partition_pdf

from backend.utils.elements import Image, Table, Text
from backend.utils.llm import get_text_llm, get_vision_llm
from backend.utils.retriever import add_documents_multivector, get_retriever
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

from . import prompts
from .config import Config


async def apply_summarize_text(text_list: list[Text], config) -> None:
    str_list = [text.text for text in text_list]

    if config.ingest.summarize_text:
        model = get_text_llm(config)

        text_summaries = await generate_text_summaries(
            str_list, prompt_template=prompts.TEXT_SUMMARIZATION_PROMPT, model=model
        )

    else:
        text_summaries = str_list

    for text in text_list:
        text.set_summary(text_summaries.pop(0))
    return


async def apply_summarize_table(table_list: list[Table], config) -> None:
    table_format = config.ingest.table_format
    if table_format in ["text", "html"]:
        str_list = [table.text for table in table_list]
        model = get_text_llm(config)

        table_summaries = await generate_text_summaries(
            str_list, prompt_template=prompts.TABLE_SUMMARIZATION_PROMPT, model=model
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

    return


async def apply_summarize_image(image_list: list[Image], config) -> None:
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


async def ingest_pdf(file_path, config):
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

    # Summarize text
    await apply_summarize_text(texts, config)

    # Summarize tables
    await apply_summarize_table(tables, config)

    # Summarize images
    await apply_summarize_image(images, config)

    retriever = get_retriever(config)

    # Add texts to retriever
    text_summaries = [text.get_summary() for text in texts]
    text_contents = [text.get_content() for text in texts]
    text_metadata = [text.get_metadata() for text in texts]

    add_documents_multivector(
        retriever=retriever,
        doc_summaries=text_summaries,
        doc_contents_str=text_contents,
        doc_metadata=text_metadata,
    )

    # Add tables to retriever
    table_summaries = [table.get_summary() for table in tables]
    table_contents = [table.get_content() for table in tables]
    table_metadata = [table.get_metadata() for table in tables]

    add_documents_multivector(
        retriever=retriever,
        doc_summaries=table_summaries,
        doc_contents_str=table_contents,
        doc_metadata=table_metadata,
    )

    # Add images to retriever
    image_summaries = [image.get_summary() for image in images]
    image_contents = [image.get_content() for image in images]
    image_metadata = [image.get_metadata() for image in images]

    add_documents_multivector(
        retriever=retriever,
        doc_summaries=image_summaries,
        doc_contents_str=image_contents,
        doc_metadata=image_metadata,
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
def main(config):
    # Validate config
    cfg_obj = OmegaConf.to_object(config)
    _ = Config(**cfg_obj)

    docs_folder = Path(config.path.docs)

    for file_path in tqdm(sorted(docs_folder.glob("**/*.pdf"))):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(ingest_pdf(file_path, config))


if __name__ == "__main__":
    main()
