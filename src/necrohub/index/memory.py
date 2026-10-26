from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from necrohub.contracts.serving import Chunk, RetrievedChunk


class InMemoryVectorStore:
    """Exact cosine search with numpy over the chunks of each index version."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, tuple[Chunk, npt.NDArray[np.float32]]]] = {}

    def upsert(
        self, index_version: str, chunks: Sequence[Chunk], vectors: npt.NDArray[np.float32]
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        rows = self._rows.setdefault(index_version, {})
        for chunk, vector in zip(chunks, vectors, strict=True):
            rows[chunk.chunk_id] = (chunk, vector)
        return len(chunks)

    def search(
        self,
        index_version: str,
        vector: npt.NDArray[np.float32],
        k: int,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        rows = [
            (c, v)
            for c, v in self._rows.get(index_version, {}).values()
            if doc_id is None or c.doc_id == doc_id
        ]
        if not rows:
            return []
        matrix = np.stack([v for _, v in rows])
        sims = (matrix @ vector) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector))
        scored = [(round(float(s), 6), c) for s, (c, _) in zip(sims, rows, strict=True)]
        scored.sort(key=lambda sc: (-sc[0], sc[1].chunk_id))
        return [RetrievedChunk(chunk=c, score=s) for s, c in scored[:k]]

    def count(self, index_version: str) -> int:
        return len(self._rows.get(index_version, {}))
