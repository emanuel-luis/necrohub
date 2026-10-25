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
    """Replace every cluster of overlapping or touching spans with one [LABEL], right to left.

    The cluster covers the union of its ranges; its label is the longest span's (first on a tie).
    """
    clusters: list[list[Span]] = []
    end = -1
    for span in sorted(spans, key=lambda s: (s.start, -s.end)):
        if clusters and span.start <= end:
            clusters[-1].append(span)
            end = max(end, span.end)
        else:
            clusters.append([span])
            end = span.end
    for cluster in reversed(clusters):
        label = max(cluster, key=lambda s: s.end - s.start).label  # max keeps the first on a tie
        text = f"{text[: cluster[0].start]}[{label}]{text[max(s.end for s in cluster) :]}"
    return text
