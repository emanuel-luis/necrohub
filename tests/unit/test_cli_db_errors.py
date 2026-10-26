"""A malformed DATABASE_URL never reaches the terminal: libpq repeats it, password included."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from necrohub.cli import _error_line, app

psycopg = pytest.importorskip("psycopg")

FAKE_PASSWORD = "notarealpw" + "4242"  # built at runtime; never a real credential
BAD_DSN = f"postgresql//u:{FAKE_PASSWORD}@localhost/db"  # no ":" after the scheme: parse error


def test_error_line_hides_a_malformed_dsn() -> None:
    with pytest.raises(psycopg.Error) as info:
        psycopg.connect(BAD_DSN)
    assert FAKE_PASSWORD in str(info.value)  # the leak the helper guards against
    line = _error_line(info.value)
    assert FAKE_PASSWORD not in line
    assert "DATABASE_URL" in line


def test_error_line_keeps_other_messages() -> None:
    assert _error_line(ValueError("bad k")) == "ValueError: bad k"


def test_index_load_keeps_a_malformed_dsn_out_of_the_output_and_the_manifest(
    serving_layers: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", BAD_DSN)
    runs = tmp_path / "runs"
    args = ["index", "load", "--in", str(serving_layers / "index"), "--runs-dir", str(runs)]
    result = CliRunner().invoke(app, args)
    assert isinstance(result.exception, SystemExit)
    assert result.exit_code == 1
    assert FAKE_PASSWORD not in result.output
    manifests = list(runs.rglob("*.json"))
    assert manifests  # the failed run is still recorded
    for manifest in manifests:
        text = manifest.read_text("utf-8")
        assert FAKE_PASSWORD not in text
        assert '"status": "failed"' in text


@pytest.mark.parametrize("args", [["query", "Qual a causa da morte?"], ["serve"]])
def test_commands_do_not_print_a_malformed_dsn(
    args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NECROHUB_STORE", "pgvector")
    monkeypatch.setenv("DATABASE_URL", BAD_DSN)
    monkeypatch.setenv("NECROHUB_INDEX_VERSION", "v-test")
    monkeypatch.setenv("NECROHUB_INDEX_DIR", str(tmp_path))  # no index.json there
    result = CliRunner().invoke(app, args)
    assert isinstance(result.exception, SystemExit)  # handled, not a traceback
    assert result.exit_code == 1
    assert FAKE_PASSWORD not in result.output
    assert "sqlstate" in result.output
