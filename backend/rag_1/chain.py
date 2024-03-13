"""RAG chain for Option 1."""

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from omegaconf.dictconfig import DictConfig

from backend.utils.image import resize_base64_image
from backend.utils.llm import get_vision_llm
from backend.utils.retriever import get_retriever

from . import prompts


def split_image_text_types(docs: list[Document]) -> dict[str, list]:
    """Split base64-encoded images and texts.

    Args:
        docs (list[Document]): List of documents.

    Returns:
        dict[str, list]: Dictionary containing lists of images, mime types, and texts.
    """
    img_base64_list = []
    img_mime_type_list = []
    text_list = []
    for doc in docs:
        match doc.metadata["type"]:
            case "text":
                text_list.append(doc.page_content)
            case "image":
                img = doc.page_content
                img = resize_base64_image(img)
                img_base64_list.append(resize_base64_image(img))
                img_mime_type_list.append(doc.metadata["mime_type"])
            case "table":
                if doc.metadata["format"] == "image":
                    img = doc.page_content
                    img = resize_base64_image(img)
                    img_base64_list.append(img)
                    img_mime_type_list.append(doc.metadata["mime_type"])
                else:
                    text_list.append(doc.page_content)

    return {
        "images": img_base64_list,
        "mime_types": img_mime_type_list,
        "texts": text_list,
    }


def img_prompt_func(data_dict: dict) -> list[BaseMessage]:
    """Join the context into a single string.

    Args:
        data_dict (dict): Dictionary containing the context and question.

    Returns:
        list[BaseMessage]: List of messages to be sent to the model.
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
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


def get_chain(config: DictConfig) -> RunnableSequence:
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
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain
