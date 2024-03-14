"""Configuration schema for the RAG Option 2."""

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
class SourceConfig:
    """Configuration for the vectorstore or docstore source."""

    text: Literal["content", "summary"]
    table: Literal["content", "summary"]
    image: Literal["content", "summary"]


@dataclass(config=ConfigDict(extra="forbid"))
class IngestConfig:
    """Configuration for PDF ingestion."""

    clear_database: bool

    chunking_enable: bool
    chunking_func: HydraObject

    metadata_keys: list[str]
    table_format: Literal["text", "html", "image"]

    summarize_text: bool
    summarize_table: bool

    vectorstore_source: SourceConfig
    docstore_source: SourceConfig

    export_extracted: bool

    @root_validator(pre=True)
    def validate_fields(cls, values: dict) -> dict:
        """Various checks on the fields.

        Args:
            values (dict): Field values.

        Returns:
            dict: Validated field values.
        """
        table_format = values["table_format"]
        summarize_text = values["summarize_text"]
        summarize_table = values["summarize_table"]
        vectorstore_source = values["vectorstore_source"]
        docstore_source = values["docstore_source"]

        # Check that summary is enabled when the source is set to "summary"
        if vectorstore_source["text"] == "summary" and not summarize_text:
            raise ValueError(
                "vectorstore_source.text cannot be 'summary' when summarize_text is"
                " False"
            )
        if vectorstore_source["table"] == "summary" and not summarize_table:
            raise ValueError(
                "vectorstore_source.table cannot be 'summary' when summarize_table is"
                " False"
            )
        if docstore_source["text"] == "summary" and not summarize_text:
            raise ValueError(
                "docstore_source.text cannot be 'summary' when summarize_text is False"
            )
        if docstore_source["table"] == "summary" and not summarize_table:
            raise ValueError(
                "docstore_source.table cannot be 'summary' when summarize_table is"
                " False"
            )

        # Check that the source of vectorstore is not set to "content" when the content
        # is an image
        if vectorstore_source["image"] == "content":
            raise ValueError("vectorstore_source.image cannot be 'content'")
        if table_format == "image" and vectorstore_source["table"] == "content":
            raise ValueError(
                "vectorstore_source.table cannot be 'content' when table_format is"
                " 'image'"
            )

        # Check that the source of docstore is not set to "content" when the content
        # is an image (option 2)
        if docstore_source["image"] == "content":
            raise ValueError("docstore_source.image cannot be 'content'")
        if table_format == "image" and docstore_source["table"] == "content":
            raise ValueError(
                "docstore_source.table cannot be 'content' when table_format is"
                " 'image'"
            )

        return values


@dataclass(config=ConfigDict(extra="forbid"))
class Config:
    """Configuration for the RAG Option 2."""

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
