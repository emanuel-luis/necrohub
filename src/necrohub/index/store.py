"""The vector store interface and the checks and reader shared by the index build and its sinks."""

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from necrohub.contracts.serving import Chunk, IndexInfo, RetrievedChunk
from necrohub.preprocess.deid.regex import RegexDeidentifier
from necrohub.runs import Check, file_sha256, read_jsonl

NORM_TOLERANCE = 1e-3


class VectorStore(Protocol):
    def upsert(
        self, index_version: str, chunks: Sequence[Chunk], vectors: npt.NDArray[np.float32]
    ) -> int:
        """Insert or replace the chunks of one index version and return how many were written."""
        ...

    def search(
        self,
        index_version: str,
        vector: npt.NDArray[np.float32],
        k: int,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """The k most cosine-similar chunks, best first, ties by chunk_id."""
        ...

    def count(self, index_version: str) -> int: ...


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def derive_index_version(embedder_id: str, chunks_bytes: bytes, embeddings_bytes: bytes) -> str:
    """Content-derived: the same chunks and vectors always get the same version."""
    digest = hashlib.sha256(chunks_bytes + embeddings_bytes).hexdigest()[:12]
    return f"{slug(embedder_id)}-{digest}"


def index_checks(
    chunks: Sequence[Chunk], vectors: npt.NDArray[np.float32], dim: int
) -> list[Check]:
    """The processed -> serving boundary checks. Pass means no problem was found."""
    rows_match = len(chunks) == len(vectors)
    dim_ok = vectors.ndim == 2 and vectors.shape[1] == dim
    finite = bool(np.isfinite(vectors).all())
    norms = np.linalg.norm(vectors, axis=1) if dim_ok and finite else np.zeros(0)
    bad_norms = int((np.abs(norms - 1.0) > NORM_TOLERANCE).sum())
    detector = RegexDeidentifier()
    pii = sum(1 for c in chunks if detector.detect(c.text))
    return [
        Check(
            name="rows_match",
            passed=rows_match,
            detail=f"{len(chunks)} chunks, {len(vectors)} vectors",
        ),
        Check(
            name="dim_matches_embedder", passed=dim_ok, detail=f"shape {vectors.shape}, dim {dim}"
        ),
        Check(name="no_nan", passed=finite, detail=""),
        Check(name="unit_norms", passed=not bad_norms, detail=f"{bad_norms} row(s) off by >1e-3"),
        Check(name="no_regex_pii_in_chunks", passed=not pii, detail=f"{pii} chunk(s)"),
    ]


def load_index_dir(path: Path) -> tuple[IndexInfo, list[Chunk], npt.NDArray[np.float32]]:
    """Read an index directory and run the build's checks. Raises ValueError on any problem."""
    info = IndexInfo.model_validate_json((path / "index.json").read_text("utf-8"))
    chunks_file, vectors_file = path / "chunks.jsonl", path / "embeddings.npy"
    if file_sha256(chunks_file) != info.chunks_sha256:
        raise ValueError(f"{chunks_file.as_posix()} does not match index.json")
    if file_sha256(vectors_file) != info.embeddings_sha256:
        raise ValueError(f"{vectors_file.as_posix()} does not match index.json")
    chunks = read_jsonl(chunks_file, Chunk)
    vectors = np.load(vectors_file, allow_pickle=False).astype(np.float32, copy=False)
    failed = [c.name for c in index_checks(chunks, vectors, info.dim) if not c.passed]
    if failed or len(chunks) != info.n_chunks:
        raise ValueError(f"index {info.index_version} failed: {', '.join(failed) or 'n_chunks'}")
    return info, chunks, vectors
