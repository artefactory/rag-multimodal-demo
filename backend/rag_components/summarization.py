"""Provides functions to generate text and image summaries using OpenAI's models."""

import logging
from collections.abc import Sequence

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from tenacity import (
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@retry(
    retry=retry_if_exception_type((openai.RateLimitError, openai.BadRequestError)),
    wait=wait_exponential(multiplier=10, min=10, max=160),
    stop=stop_after_delay(300),
    before=before_log(logger, logging.INFO),
    # after=after_log(logger, logging.INFO),
)
async def generate_text_summaries(
    text_list: list[str],
    prompt_template: str,
    model: BaseChatModel,
    batch_size: int = 10,
) -> list[str]:
    """Generate summaries for a list of texts.

    Args:
        text_list (list[str]): List of texts to be summarized.
        prompt_template (str): Template used to create prompts for the language model.
        model (BaseChatModel): Language model used for generating summaries.
        batch_size (int, optional): Number of texts to process simultaneously in the API
            request. Defaults to 50.

    Returns:
        list[str]: List of summaries for the texts.
    """
    # Prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Text summary chain
    summarize_chain = (
        {"text": lambda x: x} | prompt | model | StrOutputParser()
    ).with_config({"run_name": "TextSummarization"})

    # Initialize empty summaries
    text_summaries = []

    logger.info(f"Summarizing {len(text_list)} texts")

    # Process texts in batches
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        batch_summaries = await summarize_chain.abatch(batch)
        text_summaries.extend(batch_summaries)

    return text_summaries


@retry(
    retry=retry_if_exception_type((openai.RateLimitError, openai.BadRequestError)),
    wait=wait_exponential(multiplier=10, min=10, max=160),
    stop=stop_after_delay(300),
    before=before_log(logger, logging.INFO),
    # after=after_log(logger, logging.INFO),
)
async def generate_image_summaries(
    img_base64_list: list[str],
    img_mime_type_list: list[str],
    prompt: str,
    model: BaseChatModel,
    batch_size: int = 10,
) -> list[str]:
    """Generate summaries for a list of images encoded in base64.

    Args:
        img_base64_list (list[str]): List of base64-encoded strings representing images.
        img_mime_type_list (list[str]): List of MIME types corresponding to the images.
            Example: ["image/jpeg", "image/png"]
        prompt (str): Text prompt used for generating the summaries.
        model (BaseChatModel): Language model used for generating summaries.
        batch_size (int, optional): Number of images to process simultaneously in the
            API request. Defaults to 50.

    Returns:
        list[str]: List of summaries for the images.
    """
    assert len(img_base64_list) == len(img_mime_type_list)

    def _get_messages_from_url(_dict: dict[str, str]) -> Sequence[BaseMessage]:
        img_base64, img_mime_type = _dict["img_base64"], _dict["img_mime_type"]
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img_mime_type};base64,{img_base64}",
                            "detail": "high",
                        },
                    },
                ]
            )
        ]
        return messages

    chain = (
        RunnableLambda(_get_messages_from_url) | model | StrOutputParser()
    ).with_config({"run_name": "ImageSummarization"})

    logger.info(f"Summarizing {len(img_base64_list)} images")

    # Initialize lists to store results
    image_summaries = []

    # Process images in batches
    for i in range(0, len(img_base64_list), batch_size):
        batch = [
            {
                "img_base64": img_base64,
                "img_mime_type": img_mime_type,
            }
            for img_base64, img_mime_type in zip(
                img_base64_list[i : i + batch_size],
                img_mime_type_list[i : i + batch_size],
                strict=False,
            )
        ]
        batch_summaries = await chain.abatch(batch)
        image_summaries.extend(batch_summaries)

    return image_summaries
