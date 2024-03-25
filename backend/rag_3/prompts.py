"""Prompts for RAG Option 3."""

TEXT_SUMMARIZATION_PROMPT = """You are an assistant tasked with summarizing text for \
retrieval. These summaries will be embedded and used to retrieve the raw text \
elements. Give a concise summary of the text that is well optimized for retrieval.
Text:
{text}
"""

TABLE_SUMMARIZATION_PROMPT = """You are an assistant tasked with summarizing tables \
for retrieval. These summaries will be embedded and used to retrieve the raw table \
elements. Give a concise summary of the table that is well optimized for retrieval.
Table:
{text}
"""

IMAGE_SUMMARIZATION_PROMPT = """You are an assistant tasked with summarizing images \
for retrieval. These summaries will be embedded and used to retrieve the raw image. \
Give a concise summary of the image that is well optimized for retrieval."""

RAG_PROMPT = """You will be given a mixed of text, tables, and images usually of \
charts or graphs. Use this information to provide an answer to the user question.
User-provided question:
{question}
Text and / or tables:
{text}
"""
