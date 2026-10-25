from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from necrohub.embed.chunking import text_windows, tokens_with_offsets, windows


def test_windows_golden() -> None:
    assert windows(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]
    assert windows(4, 4, 0) == [(0, 4)]
    assert windows(3, 8, 2) == [(0, 3)]
    assert windows(0, 4, 1) == []


@pytest.mark.parametrize(("size", "stride"), [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 9)])
def test_windows_rejects_bad_stride(size: int, stride: int) -> None:
    with pytest.raises(ValueError):
        windows(10, size, stride)


@given(
    n=st.integers(0, 500),
    size=st.integers(1, 50),
    data=st.data(),
)
def test_windows_cover_all(n: int, size: int, data: st.DataObject) -> None:
    stride = data.draw(st.integers(0, size - 1))
    wins = windows(n, size, stride)
    covered = {i for s, e in wins for i in range(s, e)}
    assert covered == set(range(n))
    assert all(0 < e - s <= size for s, e in wins)
    if n:
        assert wins[-1][1] == n


def test_tokens_with_offsets() -> None:
    assert tokens_with_offsets("Olá, mundo!") == [
        ("Olá", 0, 3),
        (",", 3, 4),
        ("mundo", 5, 10),
        ("!", 10, 11),
    ]


def test_text_chunks_golden(report: tuple[str, str], golden: Callable[[str, object], None]) -> None:
    name, text = report
    spans = text_windows(text, 64, 8)
    assert spans[0][0] == tokens_with_offsets(text)[0][1]
    assert spans[-1][1] == tokens_with_offsets(text)[-1][2]
    golden(f"chunking/{name}.json", spans)
