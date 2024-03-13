# RAG Multimodal Demo <!-- omit from toc -->

- [Features](#features)
  - [RAG Option 1](#rag-option-1)
  - [RAG Option 3](#rag-option-3)
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

Parameters:

- `ingest.clear_database` : Whether to clear the database before ingesting new data.
- `ingest.table_format` : How to extract table with Unstructured (`text`, `html` or `image`).
- `ingest.export_extracted` : Whether to export extracted elements in local folder.
- `metadata_keys` : Unstructured metadata to use.

### RAG Option 3

Folder: [backend/rag_3](backend/rag_3)

Method:

- Use a multimodal LLM (such as GPT-4V, LLaVA, or FUYU-8b) to produce text summaries from images.
- Embed and retrieve image summaries with a reference to the raw image.
- Pass raw images and text chunks to a multimodal LLM for answer synthesis.

Backend:

- Use the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
  with [Chroma](https://www.trychroma.com/) to store raw text and images along with their summaries for retrieval.
- Use GPT-4V for both image summarization (for retrieval) as well as final answer synthesis from join review of images and texts (or tables).

Parameters:

- `ingest.clear_database` : Whether to clear the database before ingesting new data.
- `ingest.table_format` : How to extract tables with Unstructured (`text`, `html` or `image`).
- `ingest.summarize_text` : Whether to summarize texts with an LLM or use raw texts for retrieval.
- `ingest.summarize_table` : Whether to summarize tables with LLM or use raw tables for retrieval.
- `ingest.export_extracted` : Whether to export extracted elements to a local folder.
- `metadata_keys` : Metadata keys from Unstructured to use.

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

    For RAG Option 3:

    ```bash
    make ingest_rag_3
    ```

    This command will process PDFs to extract images, text, and tables, summarize them (depending on the method),
    and store the information in the retriever for later retrieval.

2. Start the backend server locally:

```bash
make serve
```

This command will launch the backend server, allowing you to access the FastAPI documentation and playground interfaces :

- FastAPI documentation: <http://0.0.0.0:8000/docs>
- RAG 1 playground interface: <http://0.0.0.0:8000/rag-1/playground/>
- RAG 3 playground interface: <http://0.0.0.0:8000/rag-3/playground/>

## Development

To set up a development environment and install pre-commit hooks, run the following commands:

```bash
poetry install --with dev
pre-commit install
```

If Poetry is not installed, you can install it using the following instructions: [Poetry Installation](https://python-poetry.org/docs/#installing-with-pipx)
