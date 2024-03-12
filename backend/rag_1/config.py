from typing import Literal, Optional

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pydantic import BaseModel, ConfigDict
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

    metadata_keys: list[str]

    export_extracted: bool


@dataclass(config=ConfigDict(extra="forbid"))
class Config:
    name: str

    path: PathConfig

    vision_llm: HydraObject
    embedding: HydraObject
    vectorstore: HydraObject
    retriever: HydraObject

    ingest: IngestConfig


def validate_config(config: DictConfig) -> Config:
    # Resolve the DictConfig to a native Python object
    cfg_obj = OmegaConf.to_object(config)
    # Instantiate the Config class
    validated_config = Config(**cfg_obj)
    return validated_config
