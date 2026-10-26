"""Records of the serving layer: chunks, retrieval hits, answers and the index descriptor."""

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from necrohub.contracts.spans import SectionLabel


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1


def chunk_id(doc_id: str, section: SectionLabel | str, ordinal: int, embedder_id: str) -> str:
    key = f"{doc_id}|{section}|{ordinal}|{embedder_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class Chunk(_Record):
    """A window of redacted text from one section of one document."""

    chunk_id: str
    doc_id: str  # equals the case_id
    section: SectionLabel
    ordinal: int = Field(ge=0)  # window number inside the section
    text: str
    embedder_id: str


class RetrievedChunk(_Record):
    chunk: Chunk
    score: float  # cosine similarity, rounded to 6 decimals


class Answer(_Record):
    question: str
    answer: str
    citations: list[Chunk]
    backend: str  # "extractive" or the LLM's name
    index_version: str


class IndexInfo(_Record):
    """index.json: what the index directory holds. The version is derived from its content."""

    index_version: str
    embedder_id: str
    dim: int = Field(gt=0)
    n_chunks: int = Field(ge=0)
    chunks_sha256: str
    embeddings_sha256: str
    params: dict[str, Any]
