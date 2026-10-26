import subprocess
import sys
from typing import Any

import pytest
from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.serve import ui

ANSWER: dict[str, Any] = {
    "schema_version": 1,
    "question": "q",
    "answer": "text",
    "citations": [
        {
            "schema_version": 1,
            "chunk_id": "abc",
            "doc_id": "case-a",
            "section": "conclusao",
            "ordinal": 0,
            "text": "chunk text",
            "embedder_id": "hashing",
        }
    ],
    "backend": "extractive",
    "index_version": "v1",
}


def test_initial_question_from_a_q_link() -> None:
    assert ui.initial_question({}) == ""
    assert ui.initial_question({"q": "  Qual a causa da morte?  "}) == "Qual a causa da morte?"
    assert len(ui.initial_question({"q": "x" * 5000})) == 1000


def test_ask_payload_trims_and_clamps_k() -> None:
    assert ui.ask_payload("  hi ", 99) == {"question": "hi", "k": 20}
    assert ui.ask_payload("hi", 0)["k"] == 1


def test_answer_view_keeps_the_shown_fields() -> None:
    view = ui.answer_view(ANSWER)
    assert (view.answer, view.backend, view.index_version) == ("text", "extractive", "v1")
    assert [(c.doc_id, c.section, c.ordinal, c.text) for c in view.citations] == [
        ("case-a", "conclusao", 0, "chunk text")
    ]


def test_error_messages_are_one_line() -> None:
    assert "unreachable" in ui.error_message(None)
    assert ui.error_message(503) == "The API answered 503."
    assert ui.error_message(502, "LLM backend failed: x") == (
        "The API answered 502: LLM backend failed: x"
    )
    assert "too short" in ui.error_message(422, [{"msg": "too short"}])


def test_api_url_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NECROHUB_API_URL", raising=False)
    assert ui.api_url() == ui.DEFAULT_API_URL
    monkeypatch.setenv("NECROHUB_API_URL", "http://api:8000/")
    assert ui.api_url() == "http://api:8000"


def test_ui_command_runs_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_call(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    result = CliRunner().invoke(app, ["ui", "--port", "9000"])
    assert result.exit_code == 0
    (argv,) = calls
    assert argv[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert argv[4].endswith("ui.py")
    assert argv[5:] == [
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "9000",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "minimal",
    ]
