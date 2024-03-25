"""RAG chain for Option 3."""

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
    RunnableSerializable,
)
from omegaconf.dictconfig import DictConfig
from pydantic import BaseModel

from backend.rag_components.chain_links.rag_with_history import (
    construct_rag_with_history,
)
from backend.rag_components.chain_links.retrieve_and_format_multimodal_docs import (
    fetch_docs_chain,
)
from backend.rag_components.llm import get_vision_llm
from backend.rag_components.retriever import get_retriever

from . import prompts


def img_prompt_func(
    data_dict: dict, document_separator: str = "\n\n"
) -> list[BaseMessage]:
    r"""Join the context into a single string with images and the question.

    Args:
        data_dict (dict): Dictionary containing the context and question.
        document_separator (str, optional): _description_. Defaults to "\n\n".

    Returns:
        list[BaseMessage]: List of messages to be sent to the model.
    """
    formatted_texts = document_separator.join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for idx, image in enumerate(data_dict["context"]["images"]):
            mime_type = data_dict["context"]["mime_types"][idx]
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image}"},
            }
            messages.append(image_message)

    # Adding the text for analysis
    prompt = prompts.RAG_PROMPT
    text_message = {
        "type": "text",
        "text": prompt.format(question=data_dict["question"], text=formatted_texts),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


class Question(BaseModel):
    """Question to be answered."""

    question: str


class Response(BaseModel):
    """Response to the question."""

    response: str


def get_base_chain(config: DictConfig) -> RunnableSequence:
    """Constructs a RAG pipeline that retrieves image and text data from documents.

    The pipeline consists of the following steps:
    1. Retrieval of documents using a retriever object.
    2. Splitting of image and text data.
    3. Prompting the model with the image and text data.
    4. Generating responses using a vision language model.
    5. Parsing the string output.

    Args:
        config (DictConfig): Configuration object.

    Returns:
        RunnableSequence: RAG pipeline.
    """
    retriever = get_retriever(config)
    model = get_vision_llm(config)

    # Define the RAG pipeline
    chain = (
        {
            "context": fetch_docs_chain(retriever),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
    typed_chain = chain.with_types(input_type=str, output_type=Response)

    return typed_chain


def get_chain(config: DictConfig) -> RunnableSerializable:
    """Get the appropriate RAG pipeline based on the configuration.

    Args:
        config (DictConfig): Configuration object.

    Returns:
        RunnableSerializable: RAG pipeline.
    """
    base_chain = get_base_chain(config)
    if config.rag.enable_chat_memory:
        chain_with_mem = construct_rag_with_history(base_chain, config)
        return chain_with_mem
    return base_chain
