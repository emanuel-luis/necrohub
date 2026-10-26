import json
import re
from pathlib import Path
from typing import Any

from typer.main import get_command

from necrohub.cli import app
from necrohub.sft.export import system_prompt

ROOT = Path(__file__).parents[2]
NAMES = ("NECROHUB_REF", "BASE_MODEL", "FALLBACK_MODEL", "N_CASES", "SEED", "MAX_EVAL")


def _nb() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (ROOT / "notebooks" / "qlora_extraction.ipynb").read_text("utf-8")
    )
    return data


def _code() -> list[str]:
    return ["".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "code"]


def test_notebook_is_clean_nbformat_4() -> None:
    nb = _nb()
    assert nb["nbformat"] == 4
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_params_cell_defines_the_names() -> None:
    first = next(c for c in _code() if "NECROHUB_REF" in c)
    for name in NAMES:
        assert re.search(rf"^{name}\b", first, re.MULTILINE), name


def test_every_necrohub_command_exists() -> None:
    commands = set(get_command(app).commands)  # type: ignore[attr-defined]
    called = set(re.findall(r'necrohub\(\s*"([a-z-]+)"', "\n".join(_code())))
    assert {"generate", "build-dataset", "export-sft", "eval-extraction"} <= called
    assert called <= commands


def test_reference_corpus_file_has_the_expected_shape() -> None:
    ref = json.loads((ROOT / "docs" / "results" / "full-corpus.json").read_text("utf-8"))
    assert re.fullmatch(r"[0-9a-f]{64}", ref["datasets_sha256"])
    assert {"cases", "records", "alignment", "seed", "params_override"} <= ref.keys()


def test_modelfile_system_prompt_is_the_extraction_prompt() -> None:
    text = (ROOT / "deploy" / "ollama" / "Modelfile").read_text("utf-8")
    assert system_prompt("extraction") in text
    assert "<|im_end|>" in text and "PARAMETER temperature 0" in text
