import re

_TOKEN = re.compile(r"\w+|[^\w\s]")


def windows(n_tokens: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Half-open [start, end) windows of `size` tokens overlapping by `stride` tokens.

    Every token is covered and the last window ends at n_tokens.
    """
    if size <= 0 or stride < 0 or stride >= size:
        raise ValueError(f"need size > 0 and 0 <= stride < size, got size={size} stride={stride}")
    out: list[tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        out.append((start, end))
        if end == n_tokens:
            break
        start += size - stride
    return out


def tokens_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Word and punctuation tokens with their character offsets."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def text_windows(text: str, size: int, stride: int) -> list[tuple[int, int]]:
    """Character offsets of each token window of text."""
    tokens = tokens_with_offsets(text)
    return [(tokens[s][1], tokens[e - 1][2]) for s, e in windows(len(tokens), size, stride)]
