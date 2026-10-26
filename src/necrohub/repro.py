"""The reproducibility gate: compare a fresh dvc.lock and metrics/ with a baseline.

Data stages must come back bit for bit. Model stages (CPU float results differ across OS and
CPU, decision 17) must have the same command, params and inputs, and metrics where

- data-determined fields match exactly: record, case and window counts, steps, dataset hashes,
  base model, task, seed, and the gold support of every label (tp + fn);
- model-determined values match within a tolerance: precision, recall, F1, accuracy, losses
  (floats, rel_tol and abs_tol) and the counts tp, fp, fn and pii_spans_missed (count_tol);
- a label missing from one side's per_label counts as all zeros, so a label that only one side
  ever predicted is compared, not reported as a structure change.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DATA_STAGES: dict[str, str] = {
    "generate": "generate.json",
    "ingest": "ingest.json",
    "preprocess": "preprocess.json",
    "build_dataset": "build_dataset.json",
}
MODEL_STAGES: dict[str, str] = {
    "train_ner": "train_ner.json",
    "train_sections": "train_sections.json",
    "evaluate": "eval.json",
}


# fields fixed by the data, the params and the code, never by the model's floats
_EXACT = frozenset(
    {
        "n_records",
        "n_cases",
        "n_windows",
        "n_steps",
        "dataset_sha256",
        "datasets_sha256",
        "base_model",
        "task",
        "seed",
        "schema_version",
        "resumed_from_step",
    }
)
_COUNTS = frozenset({"tp", "fp", "fn", "pii_spans_missed"})
_ZERO_LABEL: dict[str, Any] = {
    "tp": 0,
    "fp": 0,
    "fn": 0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
}


class _Tol:
    def __init__(self, rel_tol: float, abs_tol: float, count_tol: int) -> None:
        self.rel_tol, self.abs_tol, self.count_tol = rel_tol, abs_tol, count_tol


def _diff(a: Any, b: Any, path: str, key: str, exact: bool, tol: _Tol) -> list[str]:
    exact = exact or key in _EXACT
    if isinstance(a, dict) and isinstance(b, dict):
        if key == "per_label":
            a = {k: a.get(k, _ZERO_LABEL) for k in sorted(a.keys() | b.keys())}
            b = {k: b.get(k, _ZERO_LABEL) for k in a}
        elif a.keys() != b.keys():
            return [f"{path}: key(s) only in one side: {sorted(a.keys() ^ b.keys())}"]
        out: list[str] = []
        if {"tp", "fn"} <= a.keys() and a["tp"] + a["fn"] != b["tp"] + b["fn"]:
            out.append(f"{path}: gold support {a['tp'] + a['fn']} != {b['tp'] + b['fn']}")
        for k in sorted(a):
            out += _diff(a[k], b[k], f"{path}.{k}", k, exact, tol)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: list length {len(a)} != {len(b)}"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in _diff(x, y, f"{path}[{i}]", key, exact, tol)
        ]
    if exact or type(a) is not type(b):
        return [] if type(a) is type(b) and a == b else [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, float):
        if math.isclose(a, b, rel_tol=tol.rel_tol, abs_tol=tol.abs_tol):
            return []
        return [f"{path}: {a} != {b} (rel_tol {tol.rel_tol}, abs_tol {tol.abs_tol})"]
    if isinstance(a, int) and not isinstance(a, bool) and key in _COUNTS:
        return (
            []
            if abs(a - b) <= tol.count_tol
            else [f"{path}: {a} != {b} (count_tol {tol.count_tol})"]
        )
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def compare_model_metrics(
    base: Any, new: Any, path: str, *, rel_tol: float, abs_tol: float, count_tol: int
) -> list[str]:
    """Differences between two parsed model-stage metrics files, by the module's rules."""
    return _diff(base, new, path, "", False, _Tol(rel_tol, abs_tol, count_tol))


def _deps(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {d["path"]: d for d in entry.get("deps", []) if not d["path"].startswith("models/")}


def compare_lock(base: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Differences between two parsed dvc.lock files, by the rules in the module docstring."""
    out: list[str] = []
    b_stages, n_stages = base.get("stages", {}), new.get("stages", {})
    if b_stages.keys() != n_stages.keys():
        out.append(f"dvc.lock stages differ: {sorted(b_stages.keys() ^ n_stages.keys())}")
    for name in sorted(b_stages.keys() & n_stages.keys()):
        b, n = b_stages[name], n_stages[name]
        if name in MODEL_STAGES:
            for key in ("cmd", "params"):
                if b.get(key) != n.get(key):
                    out.append(f"dvc.lock {name}: {key} differs")
            if _deps(b) != _deps(n):
                out.append(f"dvc.lock {name}: deps differ (outside models/)")
        elif b != n:
            out.append(f"dvc.lock {name}: entry differs")
    return out


def compare_metrics(
    baseline: Path,
    current: Path,
    *,
    rel_tol: float = 0.05,
    abs_tol: float = 0.05,
    count_tol: int = 2,
) -> list[str]:
    """Differences between two metrics directories, by the rules in the module docstring."""
    out: list[str] = []
    for stage, name in {**DATA_STAGES, **MODEL_STAGES}.items():
        b, n = baseline / name, current / name
        if not b.is_file() or not n.is_file():
            out.append(f"metrics/{name}: missing in {'baseline' if not b.is_file() else 'current'}")
            continue
        if stage in DATA_STAGES:
            if b.read_bytes() != n.read_bytes():
                out.append(f"metrics/{name}: differs (data stages must match bit for bit)")
        else:
            out += compare_model_metrics(
                json.loads(b.read_text("utf-8")),
                json.loads(n.read_text("utf-8")),
                f"metrics/{name}",
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                count_tol=count_tol,
            )
    return out


def _load(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text("utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def check_repro(
    baseline_lock: Path,
    baseline_metrics: Path,
    lock: Path = Path("dvc.lock"),
    metrics: Path = Path("metrics"),
    *,
    rel_tol: float = 0.05,
    abs_tol: float = 0.05,
    count_tol: int = 2,
) -> list[str]:
    """All differences between the baseline and the current lock and metrics (empty: pass)."""
    return compare_lock(_load(baseline_lock), _load(lock)) + compare_metrics(
        baseline_metrics, metrics, rel_tol=rel_tol, abs_tol=abs_tol, count_tol=count_tol
    )
