from pathlib import Path

import pytest
from typer.testing import CliRunner

from necrohub import __version__
from necrohub.cli import app
from necrohub.settings import Settings

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.output.strip() == __version__


def test_check_identifiers_exit_codes(tmp_path: Path) -> None:
    clean = tmp_path / "clean.txt"
    clean.write_text("nothing here\n", encoding="utf-8")
    assert runner.invoke(app, ["check-identifiers", str(clean)]).exit_code == 0
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("x@" + "acme.test\n", encoding="utf-8")
    assert runner.invoke(app, ["check-identifiers", str(dirty)]).exit_code == 1


def test_settings_read_the_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().device == "auto"
    monkeypatch.setenv("NECROHUB_DEVICE", "cpu")
    assert Settings().device == "cpu"
