"""Open the configured vector store for serving (no web framework needed)."""

from necrohub.contracts.serving import IndexInfo
from necrohub.index.memory import InMemoryVectorStore
from necrohub.index.store import VectorStore, load_index_dir
from necrohub.settings import Settings


class ConfigError(Exception):
    """The settings cannot produce a working app; the server must not start."""


def open_store(settings: Settings, embedder_id: str) -> tuple[VectorStore, str, IndexInfo | None]:
    """The configured store, the index version it serves and the index.json it was checked with.

    Refuses when the index directory was built with another embedder than the configured one.
    """
    index_json = settings.index_dir / "index.json"
    info: IndexInfo | None
    if settings.store == "memory":
        info, chunks, vectors = load_index_dir(settings.index_dir)
        memory = InMemoryVectorStore()
        memory.upsert(info.index_version, chunks, vectors)
        store: VectorStore = memory
    else:
        from necrohub.index.pgvector import PgVectorStore

        if settings.database_url is None:
            raise ConfigError("NECROHUB_STORE=pgvector needs DATABASE_URL")
        info = (
            IndexInfo.model_validate_json(index_json.read_text("utf-8"))
            if index_json.is_file()
            else None
        )
        store = PgVectorStore(settings.database_url)
    if info is not None and info.embedder_id != embedder_id:
        raise ConfigError(
            f"configured embedder {embedder_id} differs from the index's {info.embedder_id}"
        )
    version = settings.index_version or (info.index_version if info else None)
    if version is None:
        raise ConfigError("set NECROHUB_INDEX_VERSION (no index.json to read it from)")
    return store, version, info
