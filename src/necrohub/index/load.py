"""The index load step: index directory -> pgvector, keyed by index_version."""

from pathlib import Path

from necrohub.index.pgvector import PgVectorStore
from necrohub.index.store import load_index_dir
from necrohub.runs import Counts, FileRef, RunManifest, sha256_tree, stage_run


def run_index_load(
    index_dir: Path, store: PgVectorStore, *, runs_dir: Path = Path("runs")
) -> RunManifest:
    """Migrate, upsert the index and check that the stored count equals the file count.

    Loading the same version again changes nothing: rows are keyed by (index_version, chunk_id).
    """
    with stage_run("index_load", {"index_dir": index_dir.as_posix()}, runs_dir) as run:
        info, chunks, vectors = load_index_dir(index_dir)
        run.inputs = [FileRef(path=index_dir.as_posix(), sha256=sha256_tree(index_dir))]
        store.migrate()
        store.upsert(info.index_version, chunks, vectors)
        stored = store.count(info.index_version)
        run.counts = Counts(read=len(chunks), written=stored)
        run.check(
            "count_matches_file",
            stored == info.n_chunks,
            f"{stored} stored, {info.n_chunks} in file",
        )
        run.raise_if_failed()
    return run
