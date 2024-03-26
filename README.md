# RAG Multimodal Demo <!-- omit from toc -->

[![CI status](https://github.com/artefactory/rag-multimodal-demo/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/rag-multimodal-demo/actions/workflows/ci.yaml?query=branch%3Amain)
[![Code quality status](https://github.com/artefactory/rag-multimodal-demo/actions/workflows/quality.yaml/badge.svg)](https://github.com/artefactory/rag-multimodal-demo/actions/workflows/quality.yaml?query=branch%3Amain)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%20-blue.svg)

[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/rag-multimodal-demo/blob/main/.pre-commit-config.yaml)

- [Features](#features)
  - [RAG Option 1](#rag-option-1)
  - [RAG Option 2](#rag-option-2)
  - [RAG Option 3](#rag-option-3)
  - [Frontend](#frontend)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)

This project demonstrates a multimodal system capable of processing and summarizing different types of data,
including text, images, and tables. It utilizes a retriever to store and manage the processed information.

## Features

- Use [Unstructured](https://unstructured.io/) to parse images, text, and tables from documents (PDFs).
- Summarization of images, tables, and text documents.
- Extraction and storage of metadata for various data types.

![alt text](https://blog.langchain.dev/content/images/size/w1600/2023/10/image-22.png)

- **Option 1**: This option involves retrieving the raw image directly from the dataset and combining it with the raw table and text data. The combined raw data is then processed by a Multimodal LLM to generate an answer. This approach uses the complete, unprocessed image data in conjunction with textual information.
  - Ingestion : Multimodal embeddings
  - RAG chain : Multimodal LLM

- **Option 2**: In this option, instead of using the raw image, an image summary is retrieved. This summary, along with the raw table and text data, is fed into a Text LLM to generate an answer.
  - Ingestion : Multimodal LLM (for summarization) + Text embeddings
  - RAG chain : Text LLM

- **Option 3**: This option also retrieves an image summary, but unlike Option 2, it passes the raw image to a Multimodal LLM for synthesis along with the raw table and text data.
  - Ingestion : Multimodal LLM (for summarization) + Text embeddings
  - RAG chain : Multimodal LLM

For all options, we can choose to treat tables as text or images.

**Common parameters**:

- `ingest.clear_database` : Whether to clear the database before ingesting new data.
- `ingest.partition_pdf_func` : Parameters for Unstructured `partition_pdf` function.
- `ingest.chunking_func` : Parameters for Unstructured chunking function.
- `ingest.metadata_keys` : Unstructured metadata to use.
- `ingest.table_format` : How to extract table with Unstructured (`text`, `html` or `image`).
- `ingest.image_min_size` : Minimum relative size for images to be considered.
- `ingest.table_min_size` : Minimum relative size for tables to be considered.
- `ingest.export_extracted` : Whether to export extracted elements in local folder.

Padding around extracted images can be adjusted by specifying two environment variables `"EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD"` and `"EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD"`.

### RAG Option 1

Folder: [backend/rag_1](backend/rag_1)

Method:

- Use multimodal embeddings (such as CLIP) to embed images and text.
- Retrieve both images and text using similarity search.
- Pass raw images and text chunks to a multimodal LLM for answer synthesis

Backend:

- Use [Open Clip](https://github.com/mlfoundations/open_clip) multi-modal embeddings.
- Use [Chroma](https://www.trychroma.com/) with support for multi-modal.
- Use GPT-4V for final answer synthesis from join review of images and texts (or tables).

### RAG Option 2

Folder: [backend/rag_2](backend/rag_2)

Method:

- Use a multimodal LLM (such as GPT-4V, LLaVA, or FUYU-8b) to produce text summaries from images.
- Embed and retrieve image summaries and texts chunks.
- Pass image summaries and text chunks to a text LLM for answer synthesis.

Backend:

- Use the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
  with [Chroma](https://www.trychroma.com/) to store raw text (or tables) and images (in a docstore) along with their summaries (in a vectorstore) for retrieval.
- Use GPT-4V for image summarization.
- Use GPT-4 for final answer synthesis from join review of image summaries and texts (or tables).

**Specific parameters:**

- `ingest.summarize_text` : Whether to summarize texts with an LLM or use raw texts for retrieval.
- `ingest.summarize_table` : Whether to summarize tables with LLM or use raw tables for retrieval.
- `ingest.vectorstore_source` : The field of documents to add into the vectorstore (`content` or `summary`).
- `ingest.docstore_source` : The field of documents to add into the docstore (`content` or `summary`).

In option 2, the vectorstore and docstore must be populated with text documents (text content or summary).

### RAG Option 3

Folder: [backend/rag_3](backend/rag_3)

Method:

- Use a multimodal LLM (such as GPT-4V, LLaVA, or FUYU-8b) to produce text summaries from images.
- Embed and retrieve image summaries with a reference to the raw image.
- Pass raw images and text chunks to a multimodal LLM for answer synthesis.

Backend:

- Use the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
  with [Chroma](https://www.trychroma.com/) to store raw text (or tables) and images (in a docstore) along with their summaries (in a vectorstore) for retrieval.
- Use GPT-4V for both image summarization (for retrieval) as well as final answer synthesis from join review of images and texts (or tables).

**Specific parameters:**

- `ingest.summarize_text` : Whether to summarize texts with an LLM or use raw texts for retrieval.
- `ingest.summarize_table` : Whether to summarize tables with LLM or use raw tables for retrieval.
- `ingest.vectorstore_source` : The field of documents to add into the vectorstore (`content` or `summary`).
- `ingest.docstore_source` : The field of documents to add into the docstore (`content` or `summary`).

In option 3, the vectorstore must be populated with text documents (text content or summary) as in option 2. However, the docstore can be populated with either text or image documents.

### Frontend

The demo Streamlit comes from [skaff-rag-accelerator](https://github.com/artefactory/skaff-rag-accelerator/). Please read [documentation](https://artefactory.github.io/skaff-rag-accelerator/) for more details.

## Installation

To set up the project, ensure you have Python version between 3.10 and 3.11. Then install the dependencies using Poetry:

```bash
poetry install
```

Before running the application, you need to set up the environment variables.
Copy the `template.env` file to a new file named `.env` and fill in the necessary API keys and endpoints:

```bash
cp template.env .env
# Edit the .env file with your actual values
```

## Usage

To use the RAG Multimodal Demo, follow these steps:

1. Ingest data from PDFs and summarize the content:

    For RAG Option 1:

    ```bash
    make ingest_rag_1
    ```

    For RAG Option 2:

    ```bash
    make ingest_rag_2
    ```

    For RAG Option 3:

    ```bash
    make ingest_rag_3
    ```

    This command will process PDFs to extract images, text, and tables, summarize them (depending on the method),
    and store the information in the retriever for later retrieval.

2. Start the backend server locally:

```bash
make serve_backend
```

This command will launch the backend server, allowing you to access the FastAPI documentation and playground interfaces :

- FastAPI documentation: <http://0.0.0.0:8000/docs>
- RAG Option 1 playground interface: <http://0.0.0.0:8000/rag-1/playground/>
- RAG Option 2 playground interface: <http://0.0.0.0:8000/rag-2/playground/>
- RAG Option 3 playground interface: <http://0.0.0.0:8000/rag-3/playground/>

3. Launch the Streamlit frontend interface:

```bash
make serve_frontend
```

## Development

To set up a development environment and install pre-commit hooks, run the following commands:

```bash
poetry install --with dev
pre-commit install
```

If Poetry is not installed, you can install it using the following instructions: [Poetry Installation](https://python-poetry.org/docs/#installing-with-pipx)
