import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from necrohub.runs import (
    RunManifest,
    StageFailed,
    atomic_dir,
    canonical_json,
    params_sha256,
    read_jsonl,
    sha256_tree,
    stage_run,
    write_jsonl,
)


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text("utf-8") for p in root.rglob("*.txt")}


def test_atomic_dir_replaces_the_previous_output(tmp_path: Path) -> None:
    out = tmp_path / "raw"
    (out / "old").mkdir(parents=True)
    (out / "old" / "a.txt").write_text("old", encoding="utf-8")
    with atomic_dir(out, "r1") as tmp:
        (tmp / "b.txt").write_text("new", encoding="utf-8")
        assert tmp != out and _tree(out) == {"old/a.txt": "old"}
    assert _tree(out) == {"b.txt": "new"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw"]


def test_atomic_dir_keeps_the_previous_output_on_failure(tmp_path: Path) -> None:
    out = tmp_path / "raw"
    out.mkdir()
    (out / "a.txt").write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError), atomic_dir(out, "r1") as tmp:
        (tmp / "a.txt").write_text("half-written", encoding="utf-8")
        raise RuntimeError("boom")
    assert _tree(out) == {"a.txt": "old"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw"]


def test_atomic_dir_creates_a_missing_output(tmp_path: Path) -> None:
    out = tmp_path / "data" / "raw"
    with atomic_dir(out, "r1") as tmp:
        (tmp / "a.txt").write_text("x", encoding="utf-8")
    assert _tree(out) == {"a.txt": "x"}


def test_sha256_tree_tracks_content_and_paths(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "x.txt").write_bytes(b"1")
        (root / "y.txt").write_bytes(b"2")
    assert sha256_tree(a) == sha256_tree(b)
    (b / "y.txt").write_bytes(b"3")
    assert sha256_tree(a) != sha256_tree(b)
    (b / "y.txt").write_bytes(b"2")
    (b / "y.txt").rename(b / "z.txt")
    assert sha256_tree(a) != sha256_tree(b)


class _Row(BaseModel):
    id: str
    text: str


def test_jsonl_round_trip(tmp_path: Path) -> None:
    rows = [_Row(id="1", text="ação"), _Row(id="2", text="laudo\nnº 1")]
    path = tmp_path / "rows.jsonl"
    assert write_jsonl(path, rows) == 2
    assert read_jsonl(path, _Row) == rows
    assert path.read_bytes().count(b"\n") == 2


def test_params_hash_ignores_key_order() -> None:
    assert params_sha256({"a": 1, "b": [1, 2]}) == params_sha256({"b": [1, 2], "a": 1})
    assert canonical_json({"b": 1, "a": "é"}) == '{\n  "a": "é",\n  "b": 1\n}\n'


def _manifest(runs: Path) -> RunManifest:
    (path,) = (runs / "demo").glob("*.json")
    return RunManifest.model_validate_json(path.read_text("utf-8"))


def test_stage_run_writes_a_succeeded_manifest(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    with stage_run("demo", {"seed": 7}, runs_dir=runs) as run:
        run.counts.written = 3
        run.check("all good", True)
    manifest = _manifest(runs)
    assert manifest.status == "succeeded" and manifest.counts.written == 3
    assert manifest.started_at.endswith("Z") and manifest.finished_at is not None
    assert manifest.params == {"seed": 7} and len(manifest.run_id) == 32


def test_failed_check_keeps_the_output_and_marks_the_run(tmp_path: Path) -> None:
    runs, out = tmp_path / "runs", tmp_path / "raw"
    out.mkdir()
    (out / "a.txt").write_text("old", encoding="utf-8")
    with (
        pytest.raises(StageFailed, match="spans match"),
        stage_run("demo", {}, runs_dir=runs) as run,
        atomic_dir(out, run.run_id) as tmp,
    ):
        (tmp / "a.txt").write_text("bad", encoding="utf-8")
        run.check("spans match", False, "1 span off")
        run.raise_if_failed()
    assert _tree(out) == {"a.txt": "old"}
    manifest = _manifest(runs)
    assert manifest.status == "failed" and manifest.error is not None
    assert json.loads(manifest.model_dump_json())["checks"][0]["passed"] is False
