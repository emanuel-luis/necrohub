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
