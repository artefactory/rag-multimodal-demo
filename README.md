# RAG Multimodal Demo <!-- omit from toc -->

- [Features](#features)
  - [RAG Option 3](#rag-option-3)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)


This project demonstrates a multimodal system capable of processing and summarizing different types of data, including text, images, and tables. It utilizes a retriever to store and manage the processed information.

## Features

- Summarization of images, tables, and text documents.
- Extraction and storage of metadata for various data types.

![alt text](https://blog.langchain.dev/content/images/size/w1600/2023/10/image-22.png)

### RAG Option 3

Folder : [backend/rag_3](backend/rag_3)

- Use [Unstructured](https://unstructured.io/) to parse images, text, and tables from documents (PDFs).
- Use the [multi-vector retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector) with [Chroma](https://www.trychroma.com/) to store raw text and images along with their summaries for retrieval.
- Use GPT-4V for both image summarization (for retrieval) as well as final answer synthesis from join review of images and texts (or tables).

Parameters:
- `ingest.table_format` : How to extract tables with Unstructured (`text`, `html` or `image`).
- `ingest.summarize_text` : Whether to summarize texts with an LLM or use raw texts for retrieval.
- `ingest.summarize_table` : Whether to summarize tables with LLM or use raw tables for retrieval.
- `ingest.export_extracted` : Whether to export extracted elements to a local folder.
- `metadata_keys` : Metadata keys from Unstructured to use.

## Installation

To set up the project, ensure you have Python version between 3.10 and 3.11. Then install the dependencies using Poetry:

```{bash}
poetry install
```

Before running the application, you need to set up the environment variables. Copy the `template.env` file to a new file named `.env` and fill in the necessary API keys and endpoints:

```bash
cp template.env .env
# Edit the .env file with your actual values
```

## Usage

To use the RAG Multimodal Demo, follow these steps:

1. Ingest data from PDFs and summarize the content:

```{bash}
make ingest_rag_3
```

This command will process PDFs to extract images, text, and tables, summarize them, and store the information in the retriever for later retrieval.

2. Start the backend server locally:

```{bash}
make serve
```

After launching the app, you can interact with the system through the following URLs:
- FastAPI documentation: http://0.0.0.0:8000/docs
- RAG 3 playground interface: http://0.0.0.0:8000/rag-3/playground/

## Development

```{bash}
poetry install --with dev
pre-commit install
```
