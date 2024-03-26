"""Configuration schema for the RAG Option 1."""

from typing import Literal

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pydantic import BaseModel, ConfigDict, root_validator, validator
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

    partition_pdf_func: HydraObject

    chunking_enable: bool
    chunking_func: HydraObject

    metadata_keys: list[str]
    table_format: Literal["text", "html", "image"]
    image_min_size: list[float]
    table_min_size: list[float]

    export_extracted: bool

    @root_validator(pre=True)
    def validate_fields(cls, values: dict) -> dict:
        """Various checks on the fields.

        Args:
            values (dict): Field values.

        Returns:
            dict: Validated field values.
        """
        partition_pdf_func = values["partition_pdf_func"]
        table_format = values["table_format"]

        # Check that the table structure is to be inferred when the table format is set
        # to "html"
        if table_format == "html" and (
            "infer_table_structure" not in partition_pdf_func
            or not partition_pdf_func["infer_table_structure"]
        ):
            raise ValueError(
                "partition_pdf_func.infer_table_structure must be True when"
                " table_format is 'html'"
            )

        # Check that tables are to be extracted as images when the table format is set
        # to "image"
        if table_format == "image" and (
            "extract_image_block_types" not in partition_pdf_func
            or "table" not in partition_pdf_func["extract_image_block_types"]
        ):
            raise ValueError(
                "partition_pdf_func.extract_image_block_types must contain 'table'"
                " when table_format is 'image'"
            )

        return values

    @validator("image_min_size", "table_min_size")
    def validate_size(cls, value: list[float]) -> list[float]:
        """Check that the value is between 0 and 1."""
        if len(value) != 2:
            raise ValueError("Size must be a list of two floats.")
        if min(value) < 0 or max(value) > 1:
            raise ValueError("Size must be a list of floats between 0 and 1.")
        return value


@dataclass(config=ConfigDict(extra="forbid"))
class RagConfig:
    """Configuration for RAG."""

    database_url: str
    enable_chat_memory: bool


@dataclass(config=ConfigDict(extra="forbid"))
class Config:
    """Configuration for the RAG Option 1."""

    name: str

    path: PathConfig

    text_llm: HydraObject
    vision_llm: HydraObject
    embedding: HydraObject
    vectorstore: HydraObject
    retriever: HydraObject

    ingest: IngestConfig

    rag: RagConfig


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
