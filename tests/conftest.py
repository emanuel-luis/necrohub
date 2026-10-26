import json
from collections.abc import Callable
from pathlib import Path

import pytest

from necrohub.generate.stage import run_generate
from necrohub.index.build import run_index_build
from necrohub.ingest.stage import run_ingest
from necrohub.params import GenerateParams, Params
from necrohub.preprocess.stage import run_preprocess

TESTS = Path(__file__).parent
FIXTURE_NAMES = ["firearm_a", "firearm_b", "natural"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--update-golden", action="store_true", help="rewrite tests/golden files")


@pytest.fixture(params=FIXTURE_NAMES)
def report(request: pytest.FixtureRequest) -> tuple[str, str]:
    """(name, text) of each synthetic report fixture."""
    name: str = request.param
    return name, (TESTS / "fixtures" / "reports" / f"{name}.txt").read_text("utf-8")


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> Callable[[str, object], None]:
    """golden("suite/name.json", actual): compare with, or with --update-golden write, the file."""
    update = bool(request.config.getoption("--update-golden"))

    def check(name: str, actual: object) -> None:
        if not isinstance(actual, str):
            actual = json.dumps(actual, ensure_ascii=False, indent=2) + "\n"
        path = TESTS / "golden" / name
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(actual, encoding="utf-8", newline="\n")
        else:
            assert path.exists(), f"missing golden {name}: run pytest --update-golden"
            assert actual == path.read_text("utf-8")

    return check


SERVING_PARAMS = Params(
    seed=7,
    generate=GenerateParams(
        n_cases=6,
        realizer="template",
        holdout_templates=["t3"],
        max_reject_fraction=0.2,
        pdf_fraction=0.0,
    ),
)


@pytest.fixture(scope="session")
def serving_layers(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """<base>/{raw,interim,processed,index} from one small generate-to-index run (seed 7)."""
    base = tmp_path_factory.mktemp("serving")
    kw = {"runs_dir": base / "runs", "metrics_dir": base / "metrics"}
    run_generate(SERVING_PARAMS, base / "raw", **kw)
    run_ingest(SERVING_PARAMS.ingest, base / "raw", base / "interim", **kw)
    run_preprocess(SERVING_PARAMS.preprocess, base / "interim", base / "processed", **kw)
    run_index_build(SERVING_PARAMS, base / "processed", base / "raw" / "gold", base / "index", **kw)
    return base
