"""Utility functions for vector stores."""

from hydra.utils import instantiate
from langchain_core.vectorstores import VectorStore
from omegaconf.dictconfig import DictConfig


def get_vectorstore(config: DictConfig) -> VectorStore:
    """Instantiate and return a VectorStore object based on the provided configuration.

    Args:
        config (DictConfig): Configuration object.

    Raises:
        ValueError: If the instantiated object is not a VectorStore.

    Returns:
        VectorStore: Instance of VectorStore.
    """
    object = instantiate(config.vectorstore)
    if not isinstance(object, VectorStore):
        raise ValueError(f"Expected a VectorStore object, got {type(object)}")
    return object
