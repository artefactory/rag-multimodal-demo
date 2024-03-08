from hydra.utils import instantiate
from langchain_core.language_models.chat_models import BaseChatModel


def get_text_llm(config) -> BaseChatModel:
    return instantiate(config["text_llm"])


def get_vision_llm(config) -> BaseChatModel:
    return instantiate(config["vision_llm"])
