"""dvc.yaml must list, for each stage, every necrohub source file the stage's code imports."""

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SRC = ROOT / "src"
# the CLI command and the stage module each dvc.yaml stage runs
ENTRY = {
    "generate": "necrohub.generate.stage",
    "ingest": "necrohub.ingest.stage",
    "preprocess": "necrohub.preprocess.stage",
    "build_dataset": "necrohub.datasets.stage",
}


def _file(module: str) -> Path | None:
    base = SRC / Path(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def imported_files(module: str) -> set[Path]:
    """Source files of `module` and every necrohub module it imports, transitively, including
    imports inside functions (the stages import optional backends lazily)."""
    seen: set[Path] = set()
    todo = [module]
    while todo:
        name = todo.pop()
        path = _file(name)
        if path is None or path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            if isinstance(node, ast.Import):
                todo += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                todo.append(node.module)
                todo += [f"{node.module}.{a.name}" for a in node.names]
    seen.add(SRC / "necrohub" / "cli.py")  # runs the stage; its other commands do not matter
    return {p for p in seen if p.parts[-2:] != ("necrohub", "__init__.py")}


@pytest.mark.parametrize("stage", sorted(ENTRY))
def test_stage_deps_cover_imported_sources(stage: str) -> None:
    deps = yaml.safe_load((ROOT / "dvc.yaml").read_text("utf-8"))["stages"][stage]["deps"]
    covered = [ROOT / d for d in deps]
    missing = sorted(
        p.relative_to(ROOT).as_posix()
        for p in imported_files(ENTRY[stage])
        if not any(p == c or c in p.parents for c in covered)
    )
    assert not missing, f"{stage} deps miss {missing}"
