import json
from collections.abc import Callable
from pathlib import Path

import pytest

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
