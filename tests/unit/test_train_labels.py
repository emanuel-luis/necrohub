from hypothesis import given
from hypothesis import strategies as st

from necrohub.contracts.spans import EntityLabel, Span
from necrohub.train.labels import (
    IGNORE,
    NER_LABELS,
    SECTION_LABELS,
    bio_labels,
    decode_bio,
    merge_windows,
)

TEXT = "Dra. Ana Lima examinou o corpo em Imperatriz hoje"


def tokens(text: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for word in text.split(" "):
        out.append((pos, pos + len(word)))
        pos += len(word) + 1
    return out


def span(text: str, start: int, end: int, label: EntityLabel) -> Span:
    return Span(start=start, end=end, label=label, text=text[start:end])


def test_label_lists_are_fixed_by_the_enums() -> None:
    assert NER_LABELS[:3] == ["O", "B-PER", "I-PER"]
    assert len(NER_LABELS) == 1 + 2 * len(EntityLabel)
    assert len(SECTION_LABELS) == 8


def test_bio_special_tokens_and_continuation() -> None:
    offsets = [(0, 0), *tokens(TEXT), (0, 0)]
    spans = [span(TEXT, 5, 13, EntityLabel.PER), span(TEXT, 34, 44, EntityLabel.LOC)]
    names = [("-" if i == IGNORE else NER_LABELS[i]) for i in bio_labels(offsets, spans)]
    assert names == ["-", "O", "B-PER", "I-PER", "O", "O", "O", "O", "B-LOC", "O", "-"]
    # a window that starts inside the span continues it
    cut = bio_labels([(6, 9), (10, 13), (14, 22)], spans[:1])
    assert [NER_LABELS[i] for i in cut] == ["I-PER", "I-PER", "O"]


def test_decode_round_trip_and_stray_i() -> None:
    offsets = tokens(TEXT)
    spans = [span(TEXT, 5, 13, EntityLabel.PER), span(TEXT, 34, 44, EntityLabel.LOC)]
    assert decode_bio(offsets, bio_labels(offsets, spans), TEXT) == spans
    i_per, i_loc = NER_LABELS.index("I-PER"), NER_LABELS.index("I-LOC")
    ids = [0, i_per, i_per, 0, 0, 0, 0, i_loc, i_per]
    got = decode_bio(offsets, ids, TEXT)
    assert [(s.start, s.end, s.label) for s in got] == [
        (5, 13, EntityLabel.PER),
        (34, 44, EntityLabel.LOC),
        (45, 49, EntityLabel.PER),
    ]


@given(
    st.lists(st.integers(1, 5), min_size=1, max_size=12),
    st.lists(
        st.tuples(st.booleans(), st.sampled_from(list(EntityLabel))), min_size=12, max_size=12
    ),
)
def test_round_trip_property(widths: list[int], marks: list[tuple[bool, EntityLabel]]) -> None:
    text = " ".join("x" * w for w in widths)
    offsets = tokens(text)
    spans: list[Span] = []
    i = 0
    while i < len(offsets):
        take, label = marks[i]
        if take:
            j = min(len(offsets), i + 1 + widths[i] % 3)
            spans.append(span(text, offsets[i][0], offsets[j - 1][1], label))
            i = j + 1  # a gap token keeps spans from touching
        else:
            i += 1
    assert decode_bio(offsets, bio_labels(offsets, spans), text) == spans


def test_merge_windows_prefers_the_window_where_the_token_is_central() -> None:
    a, b, c, d = (0, 1), (2, 3), (4, 5), (6, 7)
    w1 = ([(0, 0), a, b, c, (0, 0)], [IGNORE, 1, 2, 3, IGNORE])  # c is at the right edge
    w2 = ([(0, 0), b, c, d, (0, 0)], [IGNORE, 5, 6, 7, IGNORE])  # c is central here
    offsets, labels = merge_windows([w1, w2])
    assert offsets == [a, b, c, d]
    assert labels == [1, 2, 6, 7]  # b: margin 1 in w1 beats margin 0 in w2
