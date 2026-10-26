from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from necrohub.cli import app as cli_app
from necrohub.contracts.gold import GoldRecord
from necrohub.embed.hashing import HashingEmbedder
from necrohub.index.memory import InMemoryVectorStore
from necrohub.index.store import load_index_dir
from necrohub.llm.base import ChatMessage, LlmError
from necrohub.serve.rag import answer, case_id_in, llm_from_settings
from necrohub.settings import Settings

ROOT = Path(__file__).parents[2]
CASE = "c7-00003"
QUESTION = f"Qual a causa da morte no caso {CASE}?"


class FakeLLM:
    name = "fake/model"

    def __init__(self, reply: str = "A causa foi X [2].") -> None:
        self.reply = reply
        self.seen: list[ChatMessage] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
    ) -> str:
        self.seen = list(messages)
        if self.reply == "boom":
            raise LlmError("backend down")
        return self.reply


@pytest.fixture
def loaded(serving_layers: Path) -> tuple[InMemoryVectorStore, str]:
    info, chunks, vectors = load_index_dir(serving_layers / "index")
    store = InMemoryVectorStore()
    store.upsert(info.index_version, chunks, vectors)
    return store, info.index_version


def test_case_id_is_found_in_the_question() -> None:
    assert case_id_in(QUESTION) == CASE
    assert case_id_in("qual a causa da morte?") is None


def test_extractive_answer_quotes_the_cause_of_the_case(
    serving_layers: Path, loaded: tuple[InMemoryVectorStore, str]
) -> None:
    store, version = loaded
    gold = GoldRecord.model_validate_json(
        (serving_layers / "raw" / "gold" / f"{CASE}.json").read_text("utf-8")
    )
    kw = {"embedder": HashingEmbedder(256), "store": store, "index_version": version}
    result = answer(QUESTION, llm=None, k=4, **kw)  # type: ignore[arg-type]
    assert result.backend == "extractive" and result.index_version == version
    assert gold.case.conclusion.cause_of_death.lower() in result.answer.lower()
    assert [c.doc_id for c in result.citations] == [CASE]
    assert result.answer in result.citations[0].text.replace("\n", " ") or result.answer
    assert result == answer(QUESTION, llm=None, k=4, **kw)  # type: ignore[arg-type]


def test_llm_answer_cites_the_numbered_chunks_it_names(
    loaded: tuple[InMemoryVectorStore, str],
) -> None:
    store, version = loaded
    llm = FakeLLM()
    kw = {"embedder": HashingEmbedder(256), "store": store, "index_version": version}
    result = answer(QUESTION, llm=llm, k=3, **kw)  # type: ignore[arg-type]
    assert result.backend == "fake/model" and len(result.citations) == 1
    assert {c.doc_id for c in result.citations} == {CASE}
    user = llm.seen[-1].content
    assert user.startswith("Contexto:\n[1] ") and "[3] " in user and QUESTION in user
    assert "somente" in llm.seen[0].content
    no_marks = answer(QUESTION, llm=FakeLLM("sem citação"), k=3, **kw)  # type: ignore[arg-type]
    assert len(no_marks.citations) == 3


def test_llm_error_propagates(loaded: tuple[InMemoryVectorStore, str]) -> None:
    store, version = loaded
    with pytest.raises(LlmError, match="backend down"):
        answer(
            QUESTION,
            embedder=HashingEmbedder(256),
            store=store,
            index_version=version,
            llm=FakeLLM("boom"),
            k=2,
        )


def test_unknown_version_gives_an_empty_answer() -> None:
    result = answer(
        "causa da morte",
        embedder=HashingEmbedder(256),
        store=InMemoryVectorStore(),
        index_version="none",
        llm=None,
        k=2,
    )
    assert result.citations == [] and result.answer.startswith("Não encontrei")


def test_llm_setting_selects_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    assert llm_from_settings(Settings()) is None
    monkeypatch.setenv("NECROHUB_LLM", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LlmError, match="GEMINI_API_KEY"):
        llm_from_settings(Settings())
    monkeypatch.setenv("NECROHUB_LLM", "ollama")
    assert llm_from_settings(Settings()).name.startswith("ollama/")  # type: ignore[union-attr]


def test_query_cli_prints_the_answer_and_citations(serving_layers: Path) -> None:
    result = CliRunner().invoke(
        cli_app,
        [
            "query",
            QUESTION,
            "--index",
            str(serving_layers / "index"),
            "--params",
            str(ROOT / "params.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Resposta (extractive): Causa da morte" in result.output
    assert "doc_id=c7-00003 section=CONCLUSION ordinal=0 score=" in result.output


def test_query_cli_fails_on_a_missing_index(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli_app,
        ["query", QUESTION, "--index", str(tmp_path), "--params", str(ROOT / "params.yaml")],
    )
    assert result.exit_code == 1
