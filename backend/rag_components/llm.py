"""Utility functions for instantiating language models."""

from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from omegaconf.dictconfig import DictConfig


def get_text_llm(config: DictConfig) -> BaseChatModel:
    """Instantiate and return a BaseChatModel object based on the provided config.

    Args:
        config (DictConfig): Configuration object.

    Raises:
        ValueError: If the instantiated object is not a BaseChatModel.

    Returns:
        BaseChatModel: Instance of BaseChatModel.
    """
    object = instantiate(config.text_llm)
    if not isinstance(object, BaseChatModel):
        raise ValueError(f"Expected a BaseChatModel object, got {type(object)}")
    return object


def get_vision_llm(config: DictConfig) -> BaseChatModel:
    """Instantiate and return a BaseChatModel object based on the provided config.

    Args:
        config (DictConfig): Configuration object.

    Raises:
        ValueError: If the instantiated object is not a BaseChatModel.

    Returns:
        BaseChatModel: Instance of BaseChatModel.
    """
    object = instantiate(config.vision_llm)
    if not isinstance(object, BaseChatModel):
        raise ValueError(f"Expected a BaseChatModel object, got {type(object)}")
    return object
