from typing import Literal, Optional

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pydantic import BaseModel, ConfigDict, root_validator
from pydantic.dataclasses import dataclass


class HydraObject(BaseModel):
    target: str
    partial: Optional[bool]

    class Config:
        extra = "allow"
        fields = {"target": "_target_", "partial": "_partial_"}


@dataclass(config=ConfigDict(extra="forbid"))
class PathConfig:
    docs: str
    database: str
    export_extracted: str


@dataclass(config=ConfigDict(extra="forbid"))
class IngestConfig:
    chunking_enable: bool
    chunking_func: HydraObject

    table_format: Literal["text", "html", "image"]
    summarize_text: bool
    summarize_table: bool

    metadata_keys: list[str]

    export_extracted: bool

    @root_validator(pre=False)
    def validate_table_format(cls, values):
        table_format = values.get("table_format")
        summarize_table = values.get("summarize_table")

        if table_format == "image" and not summarize_table:
            raise ValueError("summarize_table must be True for table_format=image")

        return values


@dataclass(config=ConfigDict(extra="forbid"))
class Config:
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
    # Resolve the DictConfig to a native Python object
    cfg_obj = OmegaConf.to_object(config)
    # Instantiate the Config class
    validated_config = Config(**cfg_obj)
    return validated_config
