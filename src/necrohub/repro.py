"""The reproducibility gate: compare a fresh dvc.lock and metrics/ with a baseline.

Data stages must come back bit for bit. Model stages are gated on what the data and the code
decide, and only reported on what the model's floats decide (decision 17): every elementwise op
and random draw differs at the last bit between operating systems and CPUs, so the tiny models
start from different weights on each machine.

Gated (errors):
- data stages (generate to build_dataset, and index): the dvc.lock entry and
  metrics/<stage>.json identical;
- model stages: same cmd, params and deps outside models/, and metrics with the same structure
  (a label missing from one side's per_label counts as all zeros) and the same data-determined
  fields: record, case and window counts, steps, dataset hashes, base model, task, seed, and the
  gold support of every label (tp + fn).

Reported (warnings, errors with strict_models): model-determined values beyond a tolerance:
precision, recall, F1, accuracy and losses (rel_tol, abs_tol) and the counts tp, fp, fn and
pii_spans_missed (count_tol).
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import yaml

DATA_STAGES: dict[str, str] = {
    "generate": "generate.json",
    "ingest": "ingest.json",
    "preprocess": "preprocess.json",
    "build_dataset": "build_dataset.json",
    "index": "index.json",
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
        "regex",  # the regex baseline's scores: deterministic code over the dataset text
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


class Result(NamedTuple):
    errors: list[str]  # gated: the run is not a reproduction
    warnings: list[str]  # model-determined values beyond tolerance


class _Tol:
    def __init__(self, rel_tol: float, abs_tol: float, count_tol: int) -> None:
        self.rel_tol, self.abs_tol, self.count_tol = rel_tol, abs_tol, count_tol


def _diff(a: Any, b: Any, path: str, key: str, exact: bool, tol: _Tol, out: Result) -> None:
    exact = exact or key in _EXACT
    if isinstance(a, dict) and isinstance(b, dict):
        if key == "per_label":
            a = {k: a.get(k, _ZERO_LABEL) for k in sorted(a.keys() | b.keys())}
            b = {k: b.get(k, _ZERO_LABEL) for k in a}
        elif a.keys() != b.keys():
            out.errors.append(f"{path}: key(s) only in one side: {sorted(a.keys() ^ b.keys())}")
            return
        if {"tp", "fn"} <= a.keys() and a["tp"] + a["fn"] != b["tp"] + b["fn"]:
            out.errors.append(f"{path}: gold support {a['tp'] + a['fn']} != {b['tp'] + b['fn']}")
        for k in sorted(a):
            _diff(a[k], b[k], f"{path}.{k}", k, exact, tol, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.errors.append(f"{path}: list length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _diff(x, y, f"{path}[{i}]", key, exact, tol, out)
        return
    if type(a) is not type(b):
        out.errors.append(f"{path}: {a!r} != {b!r}")
    elif exact or not isinstance(a, int | float) or isinstance(a, bool):
        if a != b:
            out.errors.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, float):
        if not math.isclose(a, b, rel_tol=tol.rel_tol, abs_tol=tol.abs_tol):
            out.warnings.append(
                f"{path}: {a} != {b} (rel_tol {tol.rel_tol}, abs_tol {tol.abs_tol})"
            )
    elif key in _COUNTS:
        if abs(a - b) > tol.count_tol:
            out.warnings.append(f"{path}: {a} != {b} (count_tol {tol.count_tol})")
    elif a != b:
        out.errors.append(f"{path}: {a!r} != {b!r}")


def compare_model_metrics(
    base: Any, new: Any, path: str, *, rel_tol: float, abs_tol: float, count_tol: int
) -> Result:
    """Gated differences and tolerance warnings between two parsed model-stage metrics files."""
    out = Result([], [])
    _diff(base, new, path, "", False, _Tol(rel_tol, abs_tol, count_tol), out)
    return out


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
) -> Result:
    """Differences between two metrics directories, by the rules in the module docstring."""
    out = Result([], [])
    for stage, name in {**DATA_STAGES, **MODEL_STAGES}.items():
        b, n = baseline / name, current / name
        if not b.is_file() or not n.is_file():
            side = "baseline" if not b.is_file() else "current"
            out.errors.append(f"metrics/{name}: missing in {side}")
        elif stage in DATA_STAGES:
            if b.read_bytes() != n.read_bytes():
                out.errors.append(f"metrics/{name}: differs (data stages must match bit for bit)")
        else:
            got = compare_model_metrics(
                json.loads(b.read_text("utf-8")),
                json.loads(n.read_text("utf-8")),
                f"metrics/{name}",
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                count_tol=count_tol,
            )
            out.errors.extend(got.errors)
            out.warnings.extend(got.warnings)
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
    strict_models: bool = False,
) -> Result:
    """Errors (gated) and warnings (model drift) between the baseline and the current lock and
    metrics. With strict_models the warnings become errors."""
    got = compare_metrics(
        baseline_metrics, metrics, rel_tol=rel_tol, abs_tol=abs_tol, count_tol=count_tol
    )
    errors = compare_lock(_load(baseline_lock), _load(lock)) + got.errors
    if strict_models:
        return Result(errors + got.warnings, [])
    return Result(errors, got.warnings)
