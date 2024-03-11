from hydra.utils import instantiate
from langchain_core.vectorstores import VectorStore


def get_vectorstore(config) -> VectorStore:
    return instantiate(config.vectorstore)
