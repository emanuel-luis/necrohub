"""The reproducibility gate: compare a fresh dvc.lock and metrics/ with a baseline.

Data stages must come back bit for bit. Model stages (CPU float results differ across OS and
CPU, decision 17) must have the same command, params and inputs, and metrics with the same
structure, the same non-float values and floats within a tolerance.
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


def _same_structure(a: Any, b: Any, path: str, rel_tol: float, abs_tol: float) -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = (
            [f"{path}: key(s) only in one side: {sorted(set(a) ^ set(b))}"]
            if a.keys() ^ b.keys()
            else []
        )
        for k in sorted(a.keys() & b.keys()):
            out += _same_structure(a[k], b[k], f"{path}.{k}", rel_tol, abs_tol)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: list length {len(a)} != {len(b)}"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in _same_structure(x, y, f"{path}[{i}]", rel_tol, abs_tol)
        ]
    if isinstance(a, float) and isinstance(b, float):
        if math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            return []
        return [f"{path}: {a} != {b} (rel_tol {rel_tol}, abs_tol {abs_tol})"]
    if type(a) is not type(b) or a != b:
        return [f"{path}: {a!r} != {b!r}"]
    return []


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
    baseline: Path, current: Path, *, rel_tol: float = 0.05, abs_tol: float = 0.05
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
            out += _same_structure(
                json.loads(b.read_text("utf-8")),
                json.loads(n.read_text("utf-8")),
                f"metrics/{name}",
                rel_tol,
                abs_tol,
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
) -> list[str]:
    """All differences between the baseline and the current lock and metrics (empty: pass)."""
    return compare_lock(_load(baseline_lock), _load(lock)) + compare_metrics(
        baseline_metrics, metrics, rel_tol=rel_tol, abs_tol=abs_tol
    )
