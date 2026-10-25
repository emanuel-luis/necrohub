from collections.abc import Sequence
from typing import Literal, Protocol

from necrohub.contracts.spans import PredictedSpan, Span


class Deidentifier(Protocol):
    name: str

    def detect(self, text: str) -> list[PredictedSpan]: ...


def merge_spans[S: Span](
    groups: Sequence[Sequence[S]], how: Literal["longest"] = "longest"
) -> list[S]:
    """Flatten the groups and drop overlaps, keeping the longest span (the first one on a tie).

    Returns the same span objects, ordered by position; the inputs are never changed.
    """
    if how != "longest":
        raise ValueError(f"unknown merge strategy: {how}")
    candidates = sorted((s for g in groups for s in g), key=lambda s: s.start - s.end)
    kept: list[S] = []
    for span in candidates:
        if all(span.end <= k.start or span.start >= k.end for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)


def redact(text: str, spans: Sequence[Span]) -> str:
    """Replace every span with [LABEL] in one right-to-left pass. Overlaps are merged first."""
    for span in reversed(merge_spans([spans])):
        text = f"{text[: span.start]}[{span.label}]{text[span.end :]}"
    return text
