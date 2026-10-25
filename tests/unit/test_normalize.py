from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from necrohub.preprocess.normalize import DEFAULT_STEPS, normalize


def test_default_chain_golden(
    report: tuple[str, str], golden: Callable[[str, object], None]
) -> None:
    name, text = report
    out = normalize(text)
    golden(f"normalize/{name}.txt", out)
    assert normalize(out) == out


def test_preserves_newlines_and_accents() -> None:
    text = "Conclusão: Choque Hipovolêmico\n\nSão Luís/MA"
    assert normalize(text) == text


def test_fix_text_repairs_mojibake() -> None:
    original = "Instituto Médico-Legal, São Luís"
    broken = original.encode("utf-8").decode("latin-1")
    assert broken != original
    assert normalize(broken) == original


def test_crlf_and_nbsp() -> None:
    assert normalize("a\r\nb\rc d\n") == "a\nb\nc d\n"


def test_trailing_ws_and_blank_lines() -> None:
    assert normalize("a  \t\n\n\n\n\nb \n") == "a\n\nb\n"


def test_unknown_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="bogus"):
        normalize("x", [*DEFAULT_STEPS, "bogus"])


def test_idempotent_when_a_later_step_changes_ftfy_context() -> None:
    # nbsp before a mojibake-looking pair: fix_text leaves it, the nbsp step makes it a space,
    # and fix_text would then repair the pair on a second call
    once = normalize("\xa0\xbb\xa2")
    assert normalize(once) == once


@given(st.text())
def test_idempotent(text: str) -> None:
    once = normalize(text)
    assert normalize(once) == once
