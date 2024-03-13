"""Configuration schema for the RAG Option 3."""

from typing import Literal

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pydantic import BaseModel, ConfigDict, root_validator
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

    table_format: Literal["text", "html", "image"]
    summarize_text: bool
    summarize_table: bool

    metadata_keys: list[str]

    export_extracted: bool

    @root_validator(pre=False)
    def validate_table_format(cls, values: dict) -> dict:
        """Validate the 'table_format' field in relation to 'summarize_table'.

        This validator ensures that if the 'table_format' is set to 'image',
        then 'summarize_table' must also be set to True. It enforces the rule
        that image tables require summarization.

        Args:
            values (dict): Dictionnary of field values for the IngestConfig class.

        Raises:
            ValueError: If 'table_format' is 'image' and 'summarize_table' is not True.

        Returns:
            dict: The validated field values.
        """
        table_format = values.get("table_format")
        summarize_table = values.get("summarize_table")

        if table_format == "image" and not summarize_table:
            raise ValueError("summarize_table must be True for table_format=image")

        return values


@dataclass(config=ConfigDict(extra="forbid"))
class Config:
    """Configuration for the RAG Option 3."""

    name: str

    path: PathConfig

    text_llm: HydraObject
    vision_llm: HydraObject
    embedding: HydraObject
    vectorstore: HydraObject
    store: HydraObject
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
