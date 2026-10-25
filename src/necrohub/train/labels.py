"""Label sets and BIO encoding of spans over tokenizer offsets. Pure: no torch, no transformers."""

from collections.abc import Sequence

from necrohub.contracts.spans import EntityLabel, SectionLabel, Span

# fixed by the enums, never by the data, so every checkpoint shares one label order
NER_LABELS: list[str] = ["O"] + [f"{p}-{label}" for label in EntityLabel for p in ("B", "I")]
SECTION_LABELS: list[SectionLabel] = list(SectionLabel)

IGNORE = -100
_NER_ID = {name: i for i, name in enumerate(NER_LABELS)}

Offset = tuple[int, int]
Window = tuple[Sequence[Offset], Sequence[int]]  # token offsets and one label id per token


def _special(offset: Offset) -> bool:
    return offset == (0, 0)


def bio_labels(offsets: Sequence[Offset], spans: Sequence[Span]) -> list[int]:
    """One label id per token. A token is B-x when it is the first token of span x that starts
    inside x, I-x for a later token starting inside x, O otherwise; special tokens (offset
    (0, 0)) get -100. A span that began before the first token (a window cut it) continues
    with I-x."""
    real = [o for o in offsets if not _special(o)]
    first_start = real[0][0] if real else 0
    seen: set[int] = set()
    out: list[int] = []
    for start, end in offsets:
        if _special((start, end)):
            out.append(IGNORE)
            continue
        label = "O"
        for i, span in enumerate(spans):
            if span.start <= start < span.end:
                continuation = i in seen or span.start < first_start
                label = f"{'I' if continuation else 'B'}-{span.label}"
                seen.add(i)
                break
        out.append(_NER_ID[label])
    return out


def decode_bio(offsets: Sequence[Offset], label_ids: Sequence[int], text: str) -> list[Span]:
    """Consecutive B/I tokens of one label become one span [first.start, last.end). An I without
    a preceding B of the same label starts a span. Every span is verified against the text."""
    spans: list[Span] = []
    current: tuple[int, int, str] | None = None

    def close() -> None:
        nonlocal current
        if current is not None:
            start, end, label = current
            span = Span(start=start, end=end, label=EntityLabel(label), text=text[start:end])
            span.verify(text)
            spans.append(span)
        current = None

    for (start, end), label_id in zip(offsets, label_ids, strict=True):
        if _special((start, end)) or label_id == IGNORE:
            continue
        name = NER_LABELS[label_id]
        if name == "O":
            close()
            continue
        prefix, label = name.split("-", 1)
        if prefix == "I" and current is not None and current[2] == label:
            current = (current[0], end, label)
        else:
            close()
            current = (start, end, label)
    close()
    return spans


def merge_windows(windows: Sequence[Window]) -> tuple[list[Offset], list[int]]:
    """Merge the predictions of overlapping windows of one text into one token sequence.

    A token seen in several windows takes the label of the window where it is farthest from a
    window edge (counted in real tokens; the earliest window wins a tie). Special tokens are
    dropped. The result is ordered by position.
    """
    best: dict[Offset, tuple[int, int]] = {}  # offset -> (margin, label)
    for offsets, labels in windows:
        real = [(o, lab) for o, lab in zip(offsets, labels, strict=True) if not _special(o)]
        n = len(real)
        for i, (offset, label) in enumerate(real):
            margin = min(i, n - 1 - i)
            if offset not in best or margin > best[offset][0]:
                best[offset] = (margin, label)
    merged = sorted(best)
    return merged, [best[o][1] for o in merged]
