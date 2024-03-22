"""Prompts for RAG Option 1."""

DOCUMENT_TEMPLATE = """
Document metadata:
- Filename: {filename}
- Page number: {page_number}
Document content:
###
{page_content}
###
"""

RAG_PROMPT = """You will be given a mixed of text, tables, and images usually of \
charts or graphs. Use this information to provide an answer to the user question.
User-provided question:
{question}
Text and / or tables:
{text}
"""
