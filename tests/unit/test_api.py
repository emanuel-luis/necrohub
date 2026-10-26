from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from necrohub.embed.hashing import HashingEmbedder
from necrohub.llm.base import ChatMessage, LlmError
from necrohub.serve.api import create_app
from necrohub.serve.runtime import ConfigError
from necrohub.settings import Settings

ROOT = Path(__file__).parents[2]
QUESTION = "Qual a causa da morte no caso c7-00003?"


def _client(index: Path) -> TestClient:
    settings = Settings(index_dir=index, store="memory")
    return TestClient(create_app(settings, embedder=HashingEmbedder(256)))


def test_healthz_is_ok_for_a_loaded_index(serving_layers: Path) -> None:
    body = _client(serving_layers / "index").get("/healthz")
    assert body.status_code == 200
    data = body.json()
    assert data["status"] == "ok" and data["embedder_id"] == "hashing-256"
    assert data["n_chunks"] > 0 and data["index_version"].startswith("hashing-256-")


def test_healthz_is_503_when_the_version_is_missing(serving_layers: Path) -> None:
    settings = Settings(
        index_dir=serving_layers / "index", store="memory", index_version="no-such-version"
    )
    client = TestClient(create_app(settings, embedder=HashingEmbedder(256)))
    response = client.get("/healthz")
    assert response.status_code == 503 and response.json()["n_chunks"] == 0


def test_ask_answers_with_citations(serving_layers: Path) -> None:
    response = _client(serving_layers / "index").post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "extractive" and data["question"] == QUESTION
    assert data["citations"] and data["citations"][0]["doc_id"] == "c7-00003"
    assert data["answer"].lower().startswith("causa")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "x" * 1001},
        {"question": "ok", "k": 0},
        {"question": "ok", "k": 21},
        {"question": "ok", "extra": 1},
    ],
)
def test_ask_rejects_bad_input(serving_layers: Path, payload: dict[str, object]) -> None:
    assert _client(serving_layers / "index").post("/ask", json=payload).status_code == 422


def test_ask_without_a_word_is_422(serving_layers: Path) -> None:
    assert (
        _client(serving_layers / "index").post("/ask", json={"question": "???"}).status_code == 422
    )


class _BrokenLLM:
    name = "broken"

    def complete(self, messages: Sequence[ChatMessage], **_: object) -> str:
        raise LlmError("quota exceeded")


def test_ask_maps_an_llm_error_to_502(serving_layers: Path) -> None:
    settings = Settings(index_dir=serving_layers / "index", store="memory")
    client = TestClient(create_app(settings, embedder=HashingEmbedder(256), llm=_BrokenLLM()))
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 502 and "quota exceeded" in response.json()["detail"]


def test_start_is_refused_when_the_embedder_differs(serving_layers: Path) -> None:
    settings = Settings(index_dir=serving_layers / "index", store="memory")
    with pytest.raises(ConfigError, match="hashing-64"):
        create_app(settings, embedder=HashingEmbedder(64))


def test_pgvector_store_needs_a_dsn(serving_layers: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = Settings(index_dir=serving_layers / "index", store="pgvector")
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        create_app(settings, embedder=HashingEmbedder(256))


def _params_file(path: Path, dim: int) -> Path:
    path.write_text(
        "seed: 7\n"
        "generate: {n_cases: 1, realizer: template, holdout_templates: [], max_reject_fraction: 0.2}\n"
        f"embed: {{backend: hashing, dim: {dim}}}\n",
        encoding="utf-8",
    )
    return path


def test_numbered_env_var_selects_the_params_file(
    serving_layers: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong = _params_file(tmp_path / "p.yaml", 64)
    monkeypatch.setenv("NECROHUB_PARAMS", str(wrong))
    settings = Settings(index_dir=serving_layers / "index")
    assert settings.params_path == wrong
    with pytest.raises(ConfigError, match="hashing-64"):  # the embedder comes from that file
        create_app(settings)
    monkeypatch.delenv("NECROHUB_PARAMS")
    assert Settings(params_path=wrong).params_path == wrong  # the field name still works


def test_query_uses_necrohub_params(
    serving_layers: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from necrohub.cli import app as cli_app

    monkeypatch.setenv("NECROHUB_PARAMS", str(_params_file(tmp_path / "p.yaml", 64)))
    result = CliRunner().invoke(
        cli_app, ["query", QUESTION, "--index", str(serving_layers / "index")]
    )
    assert result.exit_code == 1 and "hashing-64" in result.output


def _fake_pg(stored: str | None) -> Callable[[object, str], str | None]:
    return lambda _store, _version: stored


@pytest.mark.parametrize(
    ("stored", "message"),
    [(None, "not loaded"), ("hashing-64", "differs")],
)
def test_pgvector_checks_the_stored_embedder(
    serving_layers: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored: str | None,
    message: str,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pw@localhost/db")
    monkeypatch.setattr("necrohub.index.pgvector.PgVectorStore.index_embedder_id", _fake_pg(stored))
    settings = Settings(index_dir=serving_layers / "index", store="pgvector")
    with pytest.raises(ConfigError, match=message):
        create_app(settings, embedder=HashingEmbedder(256))


def test_pgvector_accepts_a_matching_embedder(
    serving_layers: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pw@localhost/db")
    monkeypatch.setattr(
        "necrohub.index.pgvector.PgVectorStore.index_embedder_id", _fake_pg("hashing-256")
    )
    settings = Settings(index_dir=serving_layers / "index", store="pgvector")
    assert create_app(settings, embedder=HashingEmbedder(256)).title == "necrohub"
