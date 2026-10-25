"""Project character spans from one version of a text onto another (PDF extraction, normalize)."""

from bisect import bisect_right
from collections.abc import Sequence
from difflib import SequenceMatcher

from necrohub.contracts.spans import Span


def project[S: Span](source: str, target: str, spans: Sequence[S]) -> tuple[list[S], list[S]]:
    """Map spans of `source` onto `target`; returns (projected, rejected), each in input order.

    The two texts are diffed once with difflib (autojunk off). A span is projected when its first
    and last characters both fall in `equal` regions and the mapped slice of `target` still reads
    exactly the span's text; anything else (an endpoint inside a changed region, or a change
    inside the span) is rejected, never guessed. Projected spans are copies with new offsets.

    Cost: SequenceMatcher is quadratic in the worst case and roughly linear-times-repetition on
    report text; about tens of milliseconds for a pair of 2000-character reports. Each span is then
    one binary search over the matching blocks.
    """
    blocks = SequenceMatcher(None, source, target, autojunk=False).get_matching_blocks()
    starts = [b.a for b in blocks]

    def mapped(pos: int) -> int | None:
        i = bisect_right(starts, pos) - 1
        if i < 0 or pos >= blocks[i].a + blocks[i].size:
            return None
        return blocks[i].b + pos - blocks[i].a

    projected: list[S] = []
    rejected: list[S] = []
    for span in spans:
        start, last = mapped(span.start), mapped(span.end - 1)
        if start is None or last is None or target[start : last + 1] != span.text:
            rejected.append(span)
        else:
            projected.append(span.model_copy(update={"start": start, "end": last + 1}))
    return projected, rejected
