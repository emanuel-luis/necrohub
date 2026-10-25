import time

from hypothesis import given, settings
from hypothesis import strategies as st

from necrohub.contracts.spans import EntityLabel, Span
from necrohub.preprocess.align import project
from necrohub.preprocess.normalize import normalize

TEXT = (
    "Laudo de Necropsia\n\n\n"
    "Nome: João Miguel Sousa Silva, sexo masculino.\n"
    "Conclusão: morte por asfixia mecânica, homicídio.   \n"
)


def span(text: str, piece: str, label: EntityLabel = EntityLabel.PER) -> Span:
    start = text.index(piece)
    return Span(start=start, end=start + len(piece), label=label, text=piece)


SPANS = [
    span(TEXT, "João Miguel Sousa Silva"),
    span(TEXT, "asfixia mecânica", EntityLabel.CAUSE_OF_DEATH),
    span(TEXT, "homicídio", EntityLabel.MANNER_OF_DEATH),
]


def check(target: str, spans: list[Span]) -> None:
    for s in spans:
        s.verify(target)


def test_identity_keeps_every_span() -> None:
    projected, rejected = project(TEXT, TEXT, SPANS)
    assert projected == SPANS and rejected == []


def test_whitespace_drift_shifts_spans() -> None:
    # what pypdf does to an fpdf2 rendering: blank lines and trailing spaces disappear
    target = TEXT.replace("\n\n\n", "\n").replace("   \n", "\n")
    projected, rejected = project(TEXT, target, SPANS)
    assert rejected == [] and len(projected) == 3
    check(target, projected)
    assert projected[0].start == SPANS[0].start - 2


def test_normalization_drift_is_projected() -> None:
    source = TEXT.replace("\n", "\r\n").replace("sexo", "sexo ")
    target = normalize(source)
    projected, rejected = project(source, target, [span(source, s.text, s.label) for s in SPANS])
    assert rejected == [] and len(projected) == 3
    check(target, projected)


def test_line_rewrapping_outside_spans_keeps_them() -> None:
    target = TEXT.replace("Silva, sexo", "Silva,\nsexo").replace("por asfixia", "por\nasfixia")
    projected, rejected = project(TEXT, target, SPANS)
    assert rejected == [] and len(projected) == 3
    check(target, projected)


def test_rewrapping_inside_a_span_rejects_it() -> None:
    target = TEXT.replace("Miguel Sousa", "Miguel\nSousa")
    projected, rejected = project(TEXT, target, SPANS)
    assert rejected == [SPANS[0]] and len(projected) == 2


def test_changed_character_inside_a_span_is_rejected() -> None:
    target = TEXT.replace("mecânica", "mecanica")
    projected, rejected = project(TEXT, target, SPANS)
    assert rejected == [SPANS[1]]
    check(target, projected)


def test_changed_endpoint_is_rejected() -> None:
    target = TEXT.replace("homicídio", "Homicídio")
    _, rejected = project(TEXT, target, SPANS)
    assert rejected == [SPANS[2]]


def test_insertion_before_a_span_shifts_it() -> None:
    target = "CABEÇALHO EXTRA\n" + TEXT
    projected, rejected = project(TEXT, target, SPANS)
    assert rejected == []
    assert [p.start for p in projected] == [s.start + 16 for s in SPANS]
    check(target, projected)


def test_fast_enough_for_a_report() -> None:
    source = (TEXT * 12)[:2400]
    target = source.replace("\n\n", "\n").replace("   \n", "\n")
    spans = [span(source, "João Miguel Sousa Silva")]
    t0 = time.perf_counter()
    project(source, target, spans)
    assert time.perf_counter() - t0 < 2.0


# Gaps are shorter than any span and share no character with them. difflib finds longest common
# blocks, not minimal edits, so a gap longer than a span could outrank it: that reads as a move,
# and a moved span is rejected (see test_moved_span_is_rejected).
_PIECES = st.text(alphabet="abc ção\n,.", max_size=6)


@settings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(_PIECES, _PIECES), min_size=4, max_size=4))
def test_edits_outside_spans_keep_them(edits: list[tuple[str, str]]) -> None:
    # gaps between spans are edited (old gap -> new gap); span text never changes
    words = ["XYZQ-ALF", "BETA-77", "ÔMEGA_K"]
    labels = [EntityLabel.PER, EntityLabel.FINDING, EntityLabel.LOC]
    source = target = ""
    spans = []
    for i, (old, new) in enumerate(edits):
        source += old
        target += new
        if i < len(words):
            spans.append(
                Span(
                    start=len(source),
                    end=len(source) + len(words[i]),
                    label=labels[i],
                    text=words[i],
                )
            )
            source += words[i]
            target += words[i]
    projected, rejected = project(source, target, spans)
    assert rejected == []
    assert [p.text for p in projected] == words
    check(target, projected)


def test_moved_span_is_rejected() -> None:
    word = Span(start=0, end=7, label=EntityLabel.LOC, text="ÔMEGA K")
    projected, rejected = project("ÔMEGA Kaaaaaaaa", "aaaaaaaaÔMEGA K", [word])
    assert projected == [] and rejected == [word]
