"""Needs Postgres with pgvector at DATABASE_URL (CI's test-db job); skipped otherwise."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr

from necrohub.embed.hashing import HashingEmbedder
from necrohub.index.load import run_index_load
from necrohub.index.memory import InMemoryVectorStore
from necrohub.index.store import load_index_dir

if TYPE_CHECKING:
    from necrohub.index.pgvector import PgVectorStore

pytestmark = pytest.mark.db


@pytest.fixture
def pg() -> Iterator["PgVectorStore"]:
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL is not set")
    from necrohub.index.pgvector import PgVectorStore

    store = PgVectorStore(SecretStr(dsn))
    store.migrate()
    yield store


def _clean(pg: "PgVectorStore", version: str) -> None:
    import psycopg

    with psycopg.connect(pg._dsn.get_secret_value()) as conn:
        conn.execute("DELETE FROM chunks WHERE index_version = %s", (version,))
        conn.execute("DELETE FROM index_versions WHERE index_version = %s", (version,))


def test_migrate_twice_and_load_is_idempotent(
    pg: "PgVectorStore", serving_layers: Path, tmp_path: Path
) -> None:
    info, _, _ = load_index_dir(serving_layers / "index")
    _clean(pg, info.index_version)
    pg.migrate()
    pg.migrate()
    first = run_index_load(serving_layers / "index", pg, runs_dir=tmp_path)
    again = run_index_load(serving_layers / "index", pg, runs_dir=tmp_path)
    assert first.status == again.status == "succeeded"
    assert pg.count(info.index_version) == info.n_chunks  # no duplicates after the re-load
    assert pg.count("no-such-version") == 0


def test_search_matches_the_memory_store(pg: "PgVectorStore", serving_layers: Path) -> None:
    info, chunks, vectors = load_index_dir(serving_layers / "index")
    _clean(pg, info.index_version)
    pg.upsert(info.index_version, chunks, vectors)
    memory = InMemoryVectorStore()
    memory.upsert(info.index_version, chunks, vectors)
    query = HashingEmbedder(info.dim).embed(["causa da morte projétil arma de fogo"])[0]
    for doc_id in (None, "c7-00003"):
        want = memory.search(info.index_version, query, 5, doc_id=doc_id)
        got = pg.search(info.index_version, query, 5, doc_id=doc_id)
        assert [h.chunk.chunk_id for h in got] == [h.chunk.chunk_id for h in want]
        assert [h.score for h in got] == pytest.approx([h.score for h in want], abs=1e-5)
        if doc_id:
            assert {h.chunk.doc_id for h in got} == {doc_id}


def test_the_dsn_is_not_in_the_repr() -> None:
    from necrohub.index.pgvector import PgVectorStore

    store = PgVectorStore(SecretStr("postgresql://u:hunter2@h/db"))
    assert "hunter2" not in repr(store._dsn)
