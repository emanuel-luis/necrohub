import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from necrohub.contracts.documents import ProcessedDocument
from necrohub.contracts.gold import GoldRecord
from necrohub.contracts.serving import Answer, Chunk, IndexInfo, RetrievedChunk, chunk_id
from necrohub.contracts.spans import SectionLabel
from necrohub.ids import fake_cpf
from necrohub.index.build import document_chunks, gold_leaks, gold_pii_strings, run_index_build
from necrohub.index.memory import InMemoryVectorStore
from necrohub.index.store import derive_index_version, index_checks, load_index_dir
from necrohub.params import GenerateParams, Params
from necrohub.runs import StageFailed, read_jsonl, sha256_tree

PARAMS = Params(  # the same params as the serving_layers fixture
    seed=7,
    generate=GenerateParams(
        n_cases=6,
        realizer="template",
        holdout_templates=["t3"],
        max_reject_fraction=0.2,
        pdf_fraction=0.0,
    ),
)


def _chunk(doc: str, n: int = 0, text: str = "texto") -> Chunk:
    return Chunk(
        chunk_id=chunk_id(doc, SectionLabel.HISTORY, n, "e"),
        doc_id=doc,
        section=SectionLabel.HISTORY,
        ordinal=n,
        text=text,
        embedder_id="e",
    )


def _build(layers: Path, out: Path, **kw: object) -> None:
    run_index_build(
        PARAMS,
        layers / "processed",
        layers / "raw" / "gold",
        out,
        runs_dir=out.parent / "runs",
        metrics_dir=out.parent / "metrics",
        **kw,  # type: ignore[arg-type]
    )


def test_chunk_id_is_the_documented_hash() -> None:
    want = hashlib.sha256(b"c7-00001|HISTORY|2|hashing-256").hexdigest()[:16]
    assert chunk_id("c7-00001", SectionLabel.HISTORY, 2, "hashing-256") == want


def test_contracts_forbid_extra_fields_and_roundtrip() -> None:
    with pytest.raises(ValidationError):
        Chunk.model_validate({**_chunk("c").model_dump(), "extra": 1})
    answer = Answer(
        question="q",
        answer="a",
        citations=[_chunk("c")],
        backend="extractive",
        index_version="v",
    )
    assert Answer.model_validate_json(answer.model_dump_json()) == answer
    assert RetrievedChunk(chunk=_chunk("c"), score=0.5).schema_version == 1


def test_index_version_is_content_derived() -> None:
    a = derive_index_version("hf:Org/Model:last", b"x", b"y")
    assert a == derive_index_version("hf:Org/Model:last", b"x", b"y")
    assert a.startswith("hf-org-model-last-")
    assert a != derive_index_version("hf:Org/Model:last", b"x", b"z")


def test_detected_entities_are_redacted_in_every_chunk(serving_layers: Path) -> None:
    """An entity value may only remain where the detector never saw it: the same string written
    elsewhere in its section outside every detected span (a regex miss, measured by the gold
    leak metric)."""
    docs = read_jsonl(serving_layers / "processed" / "documents.jsonl", ProcessedDocument)
    seen = 0
    for doc in docs:
        chunks = document_chunks(doc, "hashing-256", 128, 32)
        assert chunks
        for section in doc.sections:
            source = doc.text[section.start : section.end]
            for e in sorted(doc.entities, key=lambda x: -x.start):
                if section.start <= e.start < section.end:
                    source = (
                        source[: e.start - section.start] + "\0" + source[e.end - section.start :]
                    )
            mine = [c for c in chunks if c.section == section.label]
            for e in doc.entities:
                if section.start <= e.start < section.end:
                    seen += 1
                    leaked = any(e.text in c.text for c in mine)
                    assert not leaked or e.text in source, (doc.doc_id, e.label)
    assert seen


def test_chunk_ids_are_unique_and_cover_every_case(serving_layers: Path) -> None:
    chunks = read_jsonl(serving_layers / "index" / "chunks.jsonl", Chunk)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.doc_id for c in chunks} == {f"c7-{i:05d}" for i in range(6)}


def test_index_build_is_byte_identical_and_loadable(serving_layers: Path, tmp_path: Path) -> None:
    again = tmp_path / "index"
    _build(serving_layers, again)
    assert sha256_tree(again) == sha256_tree(serving_layers / "index")
    info, chunks, vectors = load_index_dir(again)
    assert info.n_chunks == len(chunks) == len(vectors) and info.dim == 256
    metrics = json.loads((tmp_path / "metrics" / "index.json").read_text("utf-8"))
    assert metrics["index_version"] == info.index_version
    assert metrics["gold_pii_strings_in_chunks"] >= 0 and metrics["n_chunks"] == len(chunks)


def test_gold_leak_metric_counts_a_planted_string(serving_layers: Path) -> None:
    gold_dir = serving_layers / "raw" / "gold"
    gold = GoldRecord.model_validate_json((gold_dir / "c7-00000.json").read_text("utf-8"))
    secret = min(gold_pii_strings(gold))
    assert gold_leaks([_chunk("c7-00000", text="x " + secret)], gold_dir) == 1
    assert gold_leaks([_chunk("c7-00000", text="nada")], gold_dir) == 0


def test_checks_catch_each_defect() -> None:
    good = np.eye(2, dtype=np.float32)
    chunks = [_chunk("a", 0), _chunk("a", 1)]
    assert all(c.passed for c in index_checks(chunks, good, 2))

    def failed(c: list[Chunk], v: npt.NDArray[np.float32], dim: int) -> set[str]:
        return {x.name for x in index_checks(c, v, dim) if not x.passed}

    assert "rows_match" in failed(chunks[:1], good, 2)
    assert "dim_matches_embedder" in failed(chunks, good, 3)
    assert "no_nan" in failed(chunks, np.array([[np.nan, 0], [0, 1]], dtype=np.float32), 2)
    assert "unit_norms" in failed(chunks, good * 2, 2)
    leaky = [_chunk("a", 0, f"CPF: {fake_cpf(random.Random(1))}")]
    unit = np.ones((1, 2), dtype=np.float32) / np.float32(2**0.5)
    assert "no_regex_pii_in_chunks" in failed(leaky, unit, 2)


class _NanEmbedder:
    id = "bad"
    dim = 4

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        return np.full((len(texts), 4), np.nan, dtype=np.float32)


def test_a_failed_check_keeps_the_previous_out(serving_layers: Path, tmp_path: Path) -> None:
    out = tmp_path / "index"
    _build(serving_layers, out)
    before = sha256_tree(out)
    with pytest.raises(StageFailed):
        _build(serving_layers, out, embedder=_NanEmbedder())
    assert sha256_tree(out) == before


def test_memory_store_order_ties_filter_and_upsert() -> None:
    store = InMemoryVectorStore()
    chunks = [_chunk("a", 0), _chunk("a", 1), _chunk("b", 0), _chunk("b", 1)]
    vectors = np.array([[1, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    vectors[3] /= np.linalg.norm(vectors[3])
    assert store.upsert("v", chunks, vectors) == 4
    assert store.upsert("v", chunks, vectors) == 4 and store.count("v") == 4  # idempotent
    query = np.array([1, 0], dtype=np.float32)
    hits = store.search("v", query, 4)
    tied = sorted([chunks[0].chunk_id, chunks[2].chunk_id])
    assert [h.chunk.chunk_id for h in hits[:2]] == tied
    assert hits[0].score == 1.0 and hits[2].chunk == chunks[3]
    only_b = store.search("v", query, 4, doc_id="b")
    assert {h.chunk.doc_id for h in only_b} == {"b"} and len(only_b) == 2
    assert store.search("other", query, 3) == [] and store.count("other") == 0


def test_info_roundtrip(serving_layers: Path) -> None:
    text = (serving_layers / "index" / "index.json").read_text("utf-8")
    assert IndexInfo.model_validate_json(text).index_version.startswith("hashing-256-")
