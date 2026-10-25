import time

from hypothesis import given, settings
from hypothesis import strategies as st

from necrohub.contracts.spans import EntityLabel, Span
from necrohub.preprocess.align import project, whitespace_drift
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


def test_rewrapping_inside_a_span_projects_the_target_slice() -> None:
    target = TEXT.replace("Miguel Sousa", "Miguel\nSousa")
    projected, rejected = project(TEXT, target, SPANS)
    assert rejected == [] and projected[0].text == "João Miguel\nSousa Silva"
    check(target, projected)
    assert whitespace_drift(SPANS, projected, rejected) == 1


def test_dropped_space_inside_a_span_projects_the_target_slice() -> None:
    target = TEXT.replace("Miguel Sousa", "MiguelSousa")
    projected, _ = project(TEXT, target, SPANS)
    assert projected[0].text == "João MiguelSousa Silva"
    check(target, projected)


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


def test_repeated_word_under_whitespace_drift_keeps_its_copy() -> None:
    source, target = "SP Silva SP  e Silva\n\na", "SP  Silva\nSP  \ne\nSilva\na"
    word = Span(start=15, end=20, label=EntityLabel.PER, text="Silva")
    projected, rejected = project(source, target, [word])
    assert rejected == [] and [(p.start, p.end) for p in projected] == [(17, 22)]


def test_same_word_twice_maps_to_two_targets() -> None:
    a = Span(start=0, end=5, label=EntityLabel.PER, text="Silva")
    b = Span(start=0, end=5, label=EntityLabel.LOC, text="Silva")
    assert project("Silva x", "Silva  x", [a, b])[0] == [a, b]  # same source range: both kept
    # two copies of a word map to two distinct targets (the collision guard never fires here)
    c = Span(start=6, end=11, label=EntityLabel.PER, text="Silva")
    assert project("Silva Silva", "Silva\nSilva", [a, c]) == (
        [a, c.model_copy(update={"start": 6, "end": 11})],
        [],
    )


_TOKENS = st.sampled_from(["SP", "Silva", "e", "de", "a", "Silva,", "nº"])
_WS = st.text(alphabet=" \n\t", max_size=3)


@settings(max_examples=300, deadline=None)
@given(
    st.lists(st.tuples(_TOKENS, _WS, _WS), min_size=2, max_size=25),
    st.lists(st.tuples(st.integers(0, 24), st.integers(0, 3)), min_size=1, max_size=6),
)
def test_whitespace_only_drift_never_moves_a_span(
    parts: list[tuple[str, str, str]], picks: list[tuple[int, int]]
) -> None:
    # source and target hold the same tokens; only the whitespace before each token differs
    # (the source always separates tokens). Spans cover runs of repeated tokens.
    source = target = ""
    src_pos: list[tuple[int, int]] = []
    tgt_pos: list[tuple[int, int]] = []
    for i, (token, ws_src, ws_tgt) in enumerate(parts):
        source += (ws_src or " ") if i else ws_src
        target += ws_tgt
        src_pos.append((len(source), len(source) + len(token)))
        tgt_pos.append((len(target), len(target) + len(token)))
        source += token
        target += token
    n = len(parts)
    runs = sorted({(i % n, min(i % n + k, n - 1)) for i, k in picks})
    spans = [
        Span(
            start=src_pos[i][0],
            end=src_pos[j][1],
            label=EntityLabel.PER,
            text=source[src_pos[i][0] : src_pos[j][1]],
        )
        for i, j in runs
    ]
    projected, rejected = project(source, target, spans)
    truth = {(src_pos[i][0], src_pos[j][1]): (tgt_pos[i][0], tgt_pos[j][1]) for i, j in runs}
    assert rejected == []  # changed whitespace inside a span projects the target slice
    for p, s in zip(projected, spans, strict=True):
        assert (p.start, p.end) == truth[(s.start, s.end)]
    check(target, projected)
