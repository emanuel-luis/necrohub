"""Run metadata and safe writes shared by every stage: manifests, atomic output swaps, hashes."""

import hashlib
import json
import shutil
import subprocess
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FileRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    records: int | None = None


class Counts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read: int = 0
    written: int = 0
    rejected: int = 0


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str = ""


class StageFailed(Exception):
    """A boundary check failed. Raised inside `atomic_dir`, it keeps the previous output."""


class RunManifest(BaseModel):
    """What one stage run read, wrote and checked. Lives in runs/<stage>/<run_id>.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    stage: str
    started_at: str
    finished_at: str | None = None
    status: Literal["running", "succeeded", "failed"] = "running"
    code_version: str
    params: dict[str, Any]
    params_sha256: str
    inputs: list[FileRef] = Field(default_factory=list)
    outputs: list[FileRef] = Field(default_factory=list)
    counts: Counts = Field(default_factory=Counts)
    rejects_path: str | None = None
    checks: list[Check] = Field(default_factory=list)
    error: str | None = None

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        """Record a boundary check and return whether it passed."""
        self.checks.append(Check(name=name, passed=passed, detail=detail))
        return passed

    def raise_if_failed(self) -> None:
        """Raise StageFailed when any recorded check failed. Call it inside `atomic_dir`."""
        failed = [c.name for c in self.checks if not c.passed]
        if failed:
            raise StageFailed(f"{self.stage}: failed check(s): {', '.join(failed)}")


def utc_now() -> str:
    """The current UTC time as ISO-8601 with a Z suffix (manifests only, never data)."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def code_version() -> str:
    """The package version plus the short git commit when git and a repository are available."""
    from necrohub import __version__

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        sha = ""
    return f"{__version__}+g{sha}" if sha else __version__


def canonical_json(obj: Any) -> str:
    """Byte-stable JSON: sorted keys, two-space indent, UTF-8 characters kept, final newline."""
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def params_sha256(params: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(params)).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    """One hash over every file under root: relative POSIX paths and contents, in sorted order."""
    h = hashlib.sha256()
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix())
    for f in files:
        h.update(f"{f.relative_to(root).as_posix()}\0{file_sha256(f)}\n".encode())
    return h.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text with LF newlines through a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Write one JSON object per line and return the number of records."""
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    lines = path.read_text("utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]


@contextmanager
def atomic_dir(out: Path, run_id: str) -> Iterator[Path]:
    """Yield an empty sibling directory and swap it in for `out` only if the block succeeds.

    On any exception the temp directory is deleted and `out` is left exactly as it was. The old
    `out` is moved aside before the swap because Windows cannot rename onto an existing directory.
    """
    tmp = out.with_name(f"{out.name}.tmp-{run_id}")
    old = out.with_name(f"{out.name}.old-{run_id}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if out.exists():
        out.rename(old)
    try:
        tmp.rename(out)
    except BaseException:
        if old.exists():
            old.rename(out)
        raise
    shutil.rmtree(old, ignore_errors=True)


@contextmanager
def stage_run(
    stage: str, params: Mapping[str, Any], runs_dir: Path = Path("runs")
) -> Iterator[RunManifest]:
    """Yield the run's manifest and write it to runs/<stage>/<run_id>.json however the run ends.

    An exception marks the run failed (with the error) and is re-raised.
    """
    manifest = RunManifest(
        run_id=uuid.uuid4().hex,
        stage=stage,
        started_at=utc_now(),
        code_version=code_version(),
        params=dict(params),
        params_sha256=params_sha256(params),
    )
    try:
        yield manifest
    except BaseException as exc:
        manifest.status = "failed"
        manifest.error = f"{type(exc).__name__}: {exc}"
        raise
    else:
        manifest.status = "failed" if any(not c.passed for c in manifest.checks) else "succeeded"
    finally:
        manifest.finished_at = utc_now()
        write_text_atomic(
            runs_dir / stage / f"{manifest.run_id}.json", manifest.model_dump_json(indent=2) + "\n"
        )
