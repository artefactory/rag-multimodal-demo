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

RAG_PROMPT = """\
As a chatbot assistant, your mission is to respond to user inquiries in a precise and \
concise manner based on the documents provided as input. It is essential to respond in \
the same language in which the question was asked. Responses must be written in a \
professional style and must demonstrate great attention to detail. Do not invent \
information. You must sift through various sources of information, disregarding any \
data that is not relevant to the query's context. Your response should integrate \
knowledge from the valid sources you have identified. Additionally, the question might \
include hypothetical or counterfactual statements. You need to recognize these and \
adjust your response to provide accurate, relevant information without being misled by \
the counterfactuals. Respond to the question only taking into account the following \
context. If no context is provided, do not answer. You may provide an answer if the \
user explicitely asked for a general answer. You may ask the user to rephrase their \
question, or their permission to answer without specific context from your own \
knowledge.
Please provide a list of the sources used at the end of your response with the \
following template :
```
Sources :
- **document title**, page x
- **document title**, page x
```

Question: {question}

Context:
{text}
"""
