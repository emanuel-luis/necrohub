import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.repro import DATA_STAGES, MODEL_STAGES, compare_lock, compare_model_metrics

LOCK: dict[str, Any] = {
    "schema": "2.0",
    "stages": {
        "generate": {"cmd": "g", "deps": [{"path": "a", "md5": "1"}], "outs": [{"md5": "x"}]},
        "train_ner": {
            "cmd": "t",
            "params": {"params.yaml": {"seed": 7}},
            "deps": [{"path": "data/datasets/ner", "md5": "2"}],
            "outs": [{"path": "models/encoder-ner", "md5": "m1"}],
        },
        "evaluate": {
            "cmd": "e",
            "deps": [{"path": "models/encoder-ner", "md5": "m1"}, {"path": "d", "md5": "3"}],
            "outs": [],
        },
    },
}


def test_lock_identical_and_model_out_hash_may_differ() -> None:
    new = copy.deepcopy(LOCK)
    new["stages"]["train_ner"]["outs"][0]["md5"] = "m2"
    new["stages"]["evaluate"]["deps"][0]["md5"] = "m2"  # a models/ dep is the previous out
    assert compare_lock(LOCK, new) == []


def test_lock_differences() -> None:
    new = copy.deepcopy(LOCK)
    new["stages"]["generate"]["outs"][0]["md5"] = "y"  # data stage: any change
    new["stages"]["train_ner"]["params"]["params.yaml"]["seed"] = 8
    new["stages"]["evaluate"]["deps"][1]["md5"] = "4"
    diffs = compare_lock(LOCK, new)
    assert len(diffs) == 3
    assert any("generate" in d for d in diffs) and any("params" in d for d in diffs)
    del new["stages"]["evaluate"]
    assert any("stages differ" in d for d in compare_lock(LOCK, new))


def write_metrics(root: Path, loss: float, steps: int = 10, extra: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in DATA_STAGES.values():
        (root / name).write_text('{"n": 1}\n', encoding="utf-8")
    for name in MODEL_STAGES.values():
        body: dict[str, Any] = {"loss": loss, "n_steps": steps, "base": "tiny-scratch"}
        if extra:
            body["more"] = 1
        (root / name).write_text(json.dumps(body), encoding="utf-8")


def run(
    tmp_path: Path, base: Path, new: Path, *flags: str, env: dict[str, str] | None = None
) -> Any:
    lock = tmp_path / "dvc.lock"
    lock.write_text(yaml.safe_dump(LOCK), encoding="utf-8")
    args = ["--baseline-lock", str(lock), "--lock", str(lock)]
    args += ["--baseline-metrics", str(base), "--metrics", str(new), *flags]
    return CliRunner().invoke(app, ["check-repro", *args], env=env)


def test_cli_passes_within_tolerance(tmp_path: Path) -> None:
    write_metrics(tmp_path / "a", 0.500)
    write_metrics(tmp_path / "b", 0.520)
    result = run(tmp_path, tmp_path / "a", tmp_path / "b")
    assert result.exit_code == 0 and "warning:" not in result.output


def test_cli_model_drift_warns_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)  # CI sets it; the plain form comes first
    write_metrics(tmp_path / "a", 0.5)
    write_metrics(tmp_path / "far", 0.9)
    result = run(tmp_path, tmp_path / "a", tmp_path / "far")
    assert result.exit_code == 0
    assert "warning: metrics/train_ner.json.loss: 0.5 != 0.9" in result.output
    assert "model-determined difference(s) beyond tolerance" in result.output
    assert "::warning::" not in result.output
    on_actions = run(tmp_path, tmp_path / "a", tmp_path / "far", env={"GITHUB_ACTIONS": "true"})
    assert "::warning::metrics/train_ner.json.loss" in on_actions.output


def test_cli_strict_models_fails_on_drift(tmp_path: Path) -> None:
    write_metrics(tmp_path / "a", 0.5)
    write_metrics(tmp_path / "far", 0.9)
    assert run(tmp_path, tmp_path / "a", tmp_path / "far", "--strict-models").exit_code == 1
    assert run(tmp_path, tmp_path / "a", tmp_path / "a", "--strict-models").exit_code == 0


def test_cli_fails_on_a_data_determined_field_or_structure_change(tmp_path: Path) -> None:
    write_metrics(tmp_path / "a", 0.5)
    write_metrics(tmp_path / "steps", 0.5, steps=11)
    write_metrics(tmp_path / "keys", 0.5, extra=True)
    for other in ("steps", "keys"):
        assert run(tmp_path, tmp_path / "a", tmp_path / other).exit_code == 1


def test_cli_fails_when_a_data_metric_changes_or_is_missing(tmp_path: Path) -> None:
    write_metrics(tmp_path / "a", 0.5)
    write_metrics(tmp_path / "b", 0.5)
    (tmp_path / "b" / "generate.json").write_text('{"n": 2}\n', encoding="utf-8")
    assert run(tmp_path, tmp_path / "a", tmp_path / "b").exit_code == 1
    (tmp_path / "b" / "generate.json").unlink()
    assert run(tmp_path, tmp_path / "a", tmp_path / "b").exit_code == 1


def model_diffs(base: Any, new: Any, **kw: Any) -> list[str]:
    tol: dict[str, Any] = {"rel_tol": 0.05, "abs_tol": 0.05, "count_tol": 2}
    got = compare_model_metrics(base, new, "m", **{**tol, **kw})
    return got.errors + got.warnings


LABEL = {"tp": 4, "fp": 1, "fn": 2, "precision": 0.8, "recall": 0.6667, "f1": 0.7273}


def test_data_determined_fields_are_exact() -> None:
    base = {"n_records": 10, "n_steps": 100, "dataset_sha256": "a", "base_model": "x"}
    assert model_diffs(base, dict(base)) == []
    for key, other in (("n_records", 11), ("n_steps", 101), ("dataset_sha256", "b")):
        assert model_diffs(base, {**base, key: other}), key
    nested = {"n_records": {"train": 5, "validation": 1}}
    assert model_diffs(nested, {"n_records": {"train": 6, "validation": 1}})


def test_regex_baseline_scores_are_exact() -> None:
    base = {"regex": {"strict": {"f1": 0.7458}, "pii_spans_missed": 3}}
    drift = {"regex": {"strict": {"f1": 0.7459}, "pii_spans_missed": 3}}
    got = compare_model_metrics(base, drift, "m", rel_tol=0.05, abs_tol=0.05, count_tol=2)
    assert got.errors and not got.warnings


def test_model_determined_values_use_the_tolerances() -> None:
    base = {"train_loss": 1.0, "pii_spans_missed": 10, "per_label": {"A": LABEL}}
    close = {"train_loss": 1.04, "pii_spans_missed": 12, "per_label": {"A": {**LABEL, "fp": 3}}}
    assert model_diffs(base, close) == []
    assert model_diffs(base, {**close, "train_loss": 1.2})
    assert model_diffs(base, {**close, "pii_spans_missed": 13})
    assert model_diffs(base, close, count_tol=0)


def test_gold_support_is_exact_even_when_tp_and_fn_move() -> None:
    base = {"per_label": {"A": LABEL}}
    moved = {"per_label": {"A": {**LABEL, "tp": 5, "fn": 1, "precision": 0.8333}}}  # support 6
    assert model_diffs(base, moved, rel_tol=0.5, abs_tol=0.5) == []
    lost = {"per_label": {"A": {**LABEL, "fn": 1}}}  # support 5: the data changed
    assert any("gold support" in d for d in model_diffs(base, lost, rel_tol=0.5, abs_tol=0.5))


def test_a_label_missing_on_one_side_counts_as_zeros() -> None:
    predicted_only = {"tp": 0, "fp": 1, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    base = {"per_label": {"A": LABEL}}
    new = {"per_label": {"A": LABEL, "B": predicted_only}}
    assert model_diffs(base, new) == []  # a stray false positive is within count_tol
    with_support = {"per_label": {"A": LABEL, "B": {**predicted_only, "tp": 1, "fp": 0}}}
    assert any("gold support" in d for d in model_diffs(base, with_support))


def test_structure_and_type_changes_are_reported() -> None:
    assert model_diffs({"a": 1.0}, {"b": 1.0})
    assert model_diffs({"val_loss": None}, {"val_loss": 1.0})
    assert model_diffs({"x": [1.0, 2.0]}, {"x": [1.0]})
