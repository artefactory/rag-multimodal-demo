from typing import Literal, Optional

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


@dataclass(config=ConfigDict(extra="forbid"))
class IngestConfig:
    chunking_enable: bool
    chunking_func: HydraObject

    summarize_text: bool
    table_format: Literal["text", "html", "image"]

    metadata_keys: list[str]


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
