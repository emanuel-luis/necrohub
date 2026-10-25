"""Project character spans from one version of a text onto another (PDF extraction, normalize)."""

from bisect import bisect_right
from collections.abc import Sequence
from difflib import SequenceMatcher

from necrohub.contracts.spans import Span


def _visible(text: str) -> tuple[str, list[int]]:
    """The text without whitespace, and the original offset of each character kept."""
    offsets = [i for i, c in enumerate(text) if not c.isspace()]
    return "".join(text[i] for i in offsets), offsets


def project[S: Span](source: str, target: str, spans: Sequence[S]) -> tuple[list[S], list[S]]:
    """Map spans of `source` onto `target`; returns (projected, rejected), each in input order.

    Whitespace is removed from both texts before the diff (difflib, autojunk off), and offsets are
    mapped back afterwards. Whitespace-only drift (PDF line breaks, dropped blank lines, trailing
    spaces) then leaves the two sequences identical, so a repeated word can never be matched to
    the wrong copy. A span is projected when its first and last non-whitespace characters both
    fall in matching blocks and the mapped slice of `target` has the same non-whitespace
    characters as the span's text. Whitespace inside a span may differ (a PDF line wrap inside a
    name): the projected span's `text` is then the target slice, so it verifies against `target`
    (`whitespace_drift` counts these). Anything else (an endpoint in a changed region, a changed
    visible character inside the span) is rejected, never guessed. Spans from different source
    offsets that land on the same target offsets are all rejected. Projected spans are copies.

    Cost: SequenceMatcher is quadratic in the worst case and roughly linear-times-repetition on
    report text; tens of milliseconds for two 2000-character reports. Each span is then two
    binary searches over the matching blocks.
    """
    src, src_at = _visible(source)
    tgt, tgt_at = _visible(target)
    blocks = SequenceMatcher(None, src, tgt, autojunk=False).get_matching_blocks()
    starts = [b.a for b in blocks]

    def mapped(pos: int) -> int | None:
        """Target offset of the visible character at source offset pos, if it matched."""
        k = bisect_right(src_at, pos) - 1  # index of pos in the visible sequence
        if k < 0 or src_at[k] != pos:
            return None
        i = bisect_right(starts, k) - 1
        if i < 0 or k >= blocks[i].a + blocks[i].size:
            return None
        return tgt_at[blocks[i].b + k - blocks[i].a]

    def target_range(span: S) -> tuple[int, int] | None:
        text = span.text
        lead, trail = len(text) - len(text.lstrip()), len(text) - len(text.rstrip())
        if lead == len(text):
            return None
        first, last = mapped(span.start + lead), mapped(span.end - 1 - trail)
        if first is None or last is None:
            return None
        start, end = first - lead, last + 1 + trail
        ok = (
            start >= 0
            and end <= len(target)
            and _visible(target[start:end])[0] == _visible(text)[0]
        )
        return (start, end) if ok else None

    ranges = [target_range(s) for s in spans]
    owners: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for span, r in zip(spans, ranges, strict=True):
        if r is not None:
            owners.setdefault(r, set()).add((span.start, span.end))

    projected: list[S] = []
    rejected: list[S] = []
    for span, r in zip(spans, ranges, strict=True):
        if r is not None and len(owners[r]) == 1:
            text = target[r[0] : r[1]]
            projected.append(span.model_copy(update={"start": r[0], "end": r[1], "text": text}))
        else:
            rejected.append(span)
    return projected, rejected


def whitespace_drift(
    spans: Sequence[Span], projected: Sequence[Span], rejected: Sequence[Span]
) -> int:
    """How many of `project`'s projected spans differ from their source span in whitespace only."""
    kept = [s for s in spans if s not in rejected]
    return sum(1 for s, p in zip(kept, projected, strict=True) if p.text != s.text)
