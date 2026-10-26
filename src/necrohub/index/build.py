"""The index stage: processed layer -> redacted chunks, vectors and index.json."""

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from necrohub.contracts.documents import ProcessedDocument
from necrohub.contracts.gold import GoldRecord
from necrohub.contracts.serving import Chunk, IndexInfo, chunk_id
from necrohub.contracts.spans import SectionLabel, Span
from necrohub.embed.base import Embedder
from necrohub.embed.chunking import text_windows
from necrohub.embed.factory import build_embedder
from necrohub.index.store import derive_index_version, index_checks
from necrohub.params import Params
from necrohub.preprocess.deid.base import redact
from necrohub.runs import (
    Counts,
    FileRef,
    RunManifest,
    atomic_dir,
    canonical_json,
    file_sha256,
    read_jsonl,
    sha256_tree,
    stage_run,
    write_jsonl,
    write_text_atomic,
)

_MIN_PII_LEN = 4
_WORD = re.compile(r"\w")


def document_chunks(
    doc: ProcessedDocument, embedder_id: str, size: int, stride: int
) -> list[Chunk]:
    """Windows of the redacted text of each section. A window without a word token is dropped.

    Each section's slice is redacted on its own with the document entities inside it, clipped to
    the slice. Ordinals count the windows of a section label across the document, so chunk ids
    stay unique even if a label repeats.
    """
    chunks: list[Chunk] = []
    ordinals: Counter[SectionLabel] = Counter()
    for section in doc.sections:
        spans = [
            Span(
                start=max(e.start, section.start) - section.start,
                end=min(e.end, section.end) - section.start,
                label=e.label,
                text=doc.text[max(e.start, section.start) : min(e.end, section.end)],
            )
            for e in doc.entities
            if e.start < section.end and e.end > section.start
        ]
        text = redact(doc.text[section.start : section.end], spans)
        for start, end in text_windows(text, size, stride):
            window = text[start:end]
            if not _WORD.search(window):
                continue
            n = ordinals[section.label]
            ordinals[section.label] += 1
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(doc.doc_id, section.label, n, embedder_id),
                    doc_id=doc.doc_id,
                    section=section.label,
                    ordinal=n,
                    text=window,
                    embedder_id=embedder_id,
                )
            )
    return chunks


def gold_pii_strings(gold: GoldRecord) -> set[str]:
    return {e.text for e in gold.entities if e.label.is_pii and len(e.text) >= _MIN_PII_LEN}


def gold_leaks(chunks: Sequence[Chunk], gold_dir: Path) -> int:
    """Chunks that hold a gold PII surface string of their own case (what the regex missed)."""
    pii: dict[str, set[str]] = {}
    n = 0
    for c in chunks:
        if c.doc_id not in pii:
            path = gold_dir / f"{c.doc_id}.json"
            pii[c.doc_id] = (
                gold_pii_strings(GoldRecord.model_validate_json(path.read_text("utf-8")))
                if path.is_file()
                else set()
            )
        n += any(s in c.text for s in pii[c.doc_id])
    return n


def run_index_build(
    params: Params,
    processed: Path,
    gold_dir: Path,
    out: Path,
    *,
    embedder: Embedder | None = None,
    runs_dir: Path = Path("runs"),
    metrics_dir: Path = Path("metrics"),
) -> RunManifest:
    """Write <out>/{chunks.jsonl, embeddings.npy, index.json} and metrics/index.json.

    `embedder` replaces the one built from params.embed (tests). A failed boundary check raises
    StageFailed, marks the manifest failed and keeps the previous out.
    """
    stage_params = {"embed": params.embed.model_dump(), "index": params.index.model_dump()}
    with stage_run("index", stage_params, runs_dir) as run:
        docs = sorted(
            read_jsonl(processed / "documents.jsonl", ProcessedDocument), key=lambda d: d.doc_id
        )
        run.inputs = [
            FileRef(path=processed.as_posix(), sha256=sha256_tree(processed)),
            FileRef(path=gold_dir.as_posix(), sha256=sha256_tree(gold_dir)),
        ]
        emb = embedder or build_embedder(params.embed)
        with atomic_dir(out, run.run_id) as tmp:
            chunks = [
                c
                for d in docs
                for c in document_chunks(
                    d, emb.id, params.index.chunk_tokens, params.index.chunk_stride
                )
            ]
            vectors = np.ascontiguousarray(
                emb.embed([c.text for c in chunks]) if chunks else np.zeros((0, emb.dim)),
                dtype=np.float32,
            )
            write_jsonl(tmp / "chunks.jsonl", chunks)
            np.save(tmp / "embeddings.npy", vectors, allow_pickle=False)
            unique = len({c.chunk_id for c in chunks}) == len(chunks)
            run.check("unique_chunk_ids", unique, "")
            for check in index_checks(chunks, vectors, emb.dim):
                run.check(check.name, check.passed, check.detail)
            chunks_bytes = (tmp / "chunks.jsonl").read_bytes()
            vectors_bytes = (tmp / "embeddings.npy").read_bytes()
            info = IndexInfo(
                index_version=derive_index_version(emb.id, chunks_bytes, vectors_bytes),
                embedder_id=emb.id,
                dim=emb.dim,
                n_chunks=len(chunks),
                chunks_sha256=file_sha256(tmp / "chunks.jsonl"),
                embeddings_sha256=file_sha256(tmp / "embeddings.npy"),
                params=stage_params,
            )
            write_text_atomic(tmp / "index.json", canonical_json(info.model_dump()))
            run.counts = Counts(read=len(docs), written=len(chunks))
            run.raise_if_failed()
            run.outputs = [
                FileRef(path=out.as_posix(), sha256=sha256_tree(tmp), records=len(chunks))
            ]

    metrics: dict[str, Any] = {
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "chunks_per_section": dict(sorted(Counter(str(c.section) for c in chunks).items())),
        "dim": emb.dim,
        "embedder_id": emb.id,
        "index_version": info.index_version,
        "gold_pii_strings_in_chunks": gold_leaks(chunks, gold_dir),
    }
    write_text_atomic(metrics_dir / "index.json", canonical_json(metrics))
    return run
