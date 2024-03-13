"""Configuration schema for the RAG Option 1."""

from typing import Literal

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass


class HydraObject(BaseModel):
    """Configuration for objects to be instantiated by Hydra."""

    target: str
    partial: bool | None

    class Config:
        """Pydantic configuration."""

        extra = "allow"
        fields = {"target": "_target_", "partial": "_partial_"}


@dataclass(config=ConfigDict(extra="forbid"))
class PathConfig:
    """Configuration for paths."""

    docs: str
    database: str
    export_extracted: str


@dataclass(config=ConfigDict(extra="forbid"))
class IngestConfig:
    """Configuration for PDF ingestion."""

    clear_database: bool

    chunking_enable: bool
    chunking_func: HydraObject

    metadata_keys: list[str]
    table_format: Literal["text", "html", "image"]

    export_extracted: bool


@dataclass(config=ConfigDict(extra="forbid"))
class Config:
    """Configuration for the RAG Option 1."""

    name: str

    path: PathConfig

    vision_llm: HydraObject
    embedding: HydraObject
    vectorstore: HydraObject
    retriever: HydraObject

    ingest: IngestConfig


def validate_config(config: DictConfig) -> Config:
    """Validate the configuration.

    Args:
        config (DictConfig): Configuration object.

    Returns:
        Config: Validated configuration object.
    """
    # Resolve the DictConfig to a native Python object
    cfg_obj = OmegaConf.to_object(config)
    # Instantiate the Config class
    validated_config = Config(**cfg_obj)
    return validated_config
