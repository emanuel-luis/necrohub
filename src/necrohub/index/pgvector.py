"""pgvector-backed store. psycopg and pgvector are imported inside methods (the `db` extra)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import SecretStr

from necrohub.contracts.serving import Chunk, RetrievedChunk
from necrohub.contracts.spans import SectionLabel

_MIGRATION = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE IF NOT EXISTS chunks (
        index_version text NOT NULL,
        chunk_id text NOT NULL,
        doc_id text NOT NULL,
        section text NOT NULL,
        ordinal int NOT NULL,
        text text NOT NULL,
        embedder_id text NOT NULL,
        embedding vector NOT NULL,
        PRIMARY KEY (index_version, chunk_id)
    )""",
    "CREATE INDEX IF NOT EXISTS chunks_version_doc ON chunks (index_version, doc_id)",
    """CREATE TABLE IF NOT EXISTS index_versions (
        index_version text PRIMARY KEY,
        embedder_id text NOT NULL,
        dim int NOT NULL,
        n_chunks int NOT NULL,
        loaded_at timestamptz NOT NULL DEFAULT now()
    )""",
)
_UPSERT = """INSERT INTO chunks
    (index_version, chunk_id, doc_id, section, ordinal, text, embedder_id, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (index_version, chunk_id) DO UPDATE SET
        doc_id = EXCLUDED.doc_id, section = EXCLUDED.section, ordinal = EXCLUDED.ordinal,
        text = EXCLUDED.text, embedder_id = EXCLUDED.embedder_id, embedding = EXCLUDED.embedding"""
_SEARCH = """SELECT chunk_id, doc_id, section, ordinal, text, embedder_id, embedding <=> %s
    FROM chunks WHERE index_version = %s {doc_filter}
    ORDER BY embedding <=> %s, chunk_id LIMIT %s"""


class PgVectorStore:
    """Exact cosine search (no ANN index) over Postgres with the pgvector extension."""

    def __init__(self, dsn: SecretStr) -> None:
        self._dsn = dsn

    def _connect(self) -> Any:
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self._dsn.get_secret_value())
        if conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone():
            register_vector(conn)
        return conn

    def migrate(self) -> None:
        """Create the extension and tables when missing; safe to run any number of times."""
        import psycopg

        with psycopg.connect(self._dsn.get_secret_value(), autocommit=True) as conn:
            for statement in _MIGRATION:
                conn.execute(statement)

    def upsert(
        self, index_version: str, chunks: Sequence[Chunk], vectors: npt.NDArray[np.float32]
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        rows = [
            (
                index_version,
                c.chunk_id,
                c.doc_id,
                c.section.value,
                c.ordinal,
                c.text,
                c.embedder_id,
                v,
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self._connect() as conn, conn.transaction(), conn.cursor() as cur:
            cur.executemany(_UPSERT, rows)
            if rows:
                cur.execute(
                    """INSERT INTO index_versions (index_version, embedder_id, dim, n_chunks)
                    VALUES (%s, %s, %s, 0) ON CONFLICT (index_version) DO NOTHING""",
                    (index_version, chunks[0].embedder_id, vectors.shape[1]),
                )
                cur.execute(
                    """UPDATE index_versions SET n_chunks =
                    (SELECT count(*) FROM chunks WHERE index_version = %s), loaded_at = now()
                    WHERE index_version = %s""",
                    (index_version, index_version),
                )
        return len(rows)

    def search(
        self,
        index_version: str,
        vector: npt.NDArray[np.float32],
        k: int,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        query = _SEARCH.format(doc_filter="AND doc_id = %s" if doc_id is not None else "")
        args: list[Any] = [vector, index_version]
        if doc_id is not None:
            args.append(doc_id)
        args += [vector, k]
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=r[0],
                    doc_id=r[1],
                    section=SectionLabel(r[2]),
                    ordinal=r[3],
                    text=r[4],
                    embedder_id=r[5],
                ),
                score=round(1.0 - float(r[6]), 6),
            )
            for r in rows
        ]

    def count(self, index_version: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT count(*) FROM chunks WHERE index_version = %s", (index_version,)
            ).fetchone()
        return int(row[0]) if row else 0
