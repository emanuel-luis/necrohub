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


def test_gpu_check_fails_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    report = {
        "torch": "2.0",
        "cuda_build": None,
        "cuda_available": False,
        "device": None,
        "capability": None,
        "arch_list": [],
        "arch_supported": False,
    }
    monkeypatch.setattr("necrohub.train.gpu.gpu_report", lambda: report)
    result = runner.invoke(app, ["gpu-check"])
    assert result.exit_code == 1 and "CUDA is not available" in result.output


def test_gpu_check_fails_on_a_missing_arch_and_passes_on_a_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "torch": "2.0",
        "cuda_build": "12.6",
        "cuda_available": True,
        "device": "GPU",
        "capability": "sm_75",
        "arch_list": ["sm_80", "sm_90"],
        "arch_supported": False,
    }
    monkeypatch.setattr("necrohub.train.gpu.gpu_report", lambda: report)
    result = runner.invoke(app, ["gpu-check"])
    assert result.exit_code == 1 and "no kernels for sm_75" in result.output
    report.update(arch_list=["sm_75"], arch_supported=True)
    result = runner.invoke(app, ["gpu-check"])
    assert result.exit_code == 0 and "gpu-check: ok" in result.output
