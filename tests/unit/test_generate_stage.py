import json
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.contracts.spans import EntityLabel, Span
from necrohub.generate import stage
from necrohub.generate.checks import unlabeled_identifiers
from necrohub.ids import cpf_check_digits
from necrohub.params import GenerateParams, Params
from necrohub.runs import StageFailed, sha256_tree

runner = CliRunner()
PARAMS = Params(
    seed=7,
    generate=GenerateParams(
        n_cases=12, realizer="template", holdout_templates=["t3"], max_reject_fraction=0.2
    ),
)


def _run(tmp: Path, name: str) -> Path:
    out = tmp / name / "raw"
    stage.run_generate(PARAMS, out, runs_dir=tmp / "runs", metrics_dir=tmp / "metrics")
    return out


def test_generate_is_byte_identical_across_runs(tmp_path: Path) -> None:
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    assert sha256_tree(a) == sha256_tree(b)
    assert len(list((a / "reports").glob("*.txt"))) == 12
    assert len(list((a / "gold").glob("*.json"))) == 12
    assert (a / "rejects.jsonl").read_text("utf-8") == ""
    metrics = json.loads((tmp_path / "metrics" / "generate.json").read_text("utf-8"))
    assert metrics["written"] == 12 and metrics["raw_sha256"] == sha256_tree(a)
    manifests = list((tmp_path / "runs" / "generate").glob("*.json"))
    assert len(manifests) == 2
    assert all(json.loads(m.read_text("utf-8"))["status"] == "succeeded" for m in manifests)


def test_cli_generate_then_check_identifiers(tmp_path: Path) -> None:
    params = Path(__file__).parents[2] / "params.yaml"
    out = tmp_path / "raw"
    args = ["generate", "--params", str(params), "--out", str(out)]
    args += ["--runs-dir", str(tmp_path / "runs"), "--metrics-dir", str(tmp_path / "metrics")]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(list((out / "gold").glob("*.json"))) == 24
    check = runner.invoke(app, ["check-identifiers", str(out)])
    assert check.exit_code == 0, check.output


def test_failed_check_keeps_the_previous_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _run(tmp_path, "a")
    before = sha256_tree(out)
    real = stage.check_raw

    def failing(raw: Path, n: int, budget: float) -> list[tuple[str, bool, str]]:
        return [*real(raw, n, budget), ("forced", False, "test")]

    monkeypatch.setattr(stage, "check_raw", failing)
    with pytest.raises(StageFailed, match="forced"):
        stage.run_generate(
            PARAMS.model_copy(update={"seed": 8}),
            out,
            runs_dir=tmp_path / "runs",
            metrics_dir=tmp_path / "metrics",
        )
    assert sha256_tree(out) == before
    assert sorted(p.name for p in out.parent.iterdir()) == ["raw"]  # no temp dir left behind
    statuses = [
        json.loads(m.read_text("utf-8"))["status"]
        for m in (tmp_path / "runs" / "generate").glob("*.json")
    ]
    assert sorted(statuses) == ["failed", "succeeded"]


def test_unlabeled_identifier_is_found_and_labelled_one_is_not() -> None:
    base = "123456780"[::-1]  # built at runtime: a checksum-valid CPF never appears as a literal
    cpf = f"{base[:3]}.{base[3:6]}.{base[6:]}-{cpf_check_digits(base)}"
    text = f"CPF nº {cpf}."
    assert unlabeled_identifiers(text, []) == ["CPF@7", "CPF@7"]  # two CPF patterns match
    span = Span(start=7, end=7 + len(cpf), label=EntityLabel.CPF, text=cpf)
    assert unlabeled_identifiers(text, [span]) == []


def test_golden_report_and_gold(tmp_path: Path, golden: Callable[[str, object], None]) -> None:
    out = _run(tmp_path, "a")
    for case_id in ("c7-00000", "c7-00004"):
        golden(f"generate/{case_id}.txt", (out / "reports" / f"{case_id}.txt").read_text("utf-8"))
        golden(f"generate/{case_id}.json", (out / "gold" / f"{case_id}.json").read_text("utf-8"))
