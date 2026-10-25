import copy
import json
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.repro import DATA_STAGES, MODEL_STAGES, compare_lock

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


def run(tmp_path: Path, base: Path, new: Path) -> int:
    lock = tmp_path / "dvc.lock"
    lock.write_text(yaml.safe_dump(LOCK), encoding="utf-8")
    args = ["--baseline-lock", str(lock), "--lock", str(lock)]
    args += ["--baseline-metrics", str(base), "--metrics", str(new)]
    return CliRunner().invoke(app, ["check-repro", *args]).exit_code


def test_cli_passes_within_tolerance(tmp_path: Path) -> None:
    write_metrics(tmp_path / "a", 0.500)
    write_metrics(tmp_path / "b", 0.520)
    assert run(tmp_path, tmp_path / "a", tmp_path / "b") == 0


def test_cli_fails_on_float_outside_tolerance_int_or_structure(tmp_path: Path) -> None:
    write_metrics(tmp_path / "a", 0.5)
    write_metrics(tmp_path / "far", 0.9)
    write_metrics(tmp_path / "steps", 0.5, steps=11)
    write_metrics(tmp_path / "keys", 0.5, extra=True)
    for other in ("far", "steps", "keys"):
        assert run(tmp_path, tmp_path / "a", tmp_path / other) == 1


def test_cli_fails_when_a_data_metric_changes_or_is_missing(tmp_path: Path) -> None:
    write_metrics(tmp_path / "a", 0.5)
    write_metrics(tmp_path / "b", 0.5)
    (tmp_path / "b" / "generate.json").write_text('{"n": 2}\n', encoding="utf-8")
    assert run(tmp_path, tmp_path / "a", tmp_path / "b") == 1
    (tmp_path / "b" / "generate.json").unlink()
    assert run(tmp_path, tmp_path / "a", tmp_path / "b") == 1
