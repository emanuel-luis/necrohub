from necrohub.contracts.spans import EntityLabel, PredictedSpan
from necrohub.preprocess.deid.base import merge_spans, redact


def span(start: int, end: int, label: EntityLabel, text: str = "") -> PredictedSpan:
    return PredictedSpan(
        start=start,
        end=end,
        label=label,
        text=text or "x" * (end - start),
        score=1.0,
        source="regex",
    )


def test_longest_wins() -> None:
    short = span(2, 5, EntityLabel.RG)
    long = span(0, 8, EntityLabel.PER)
    assert merge_spans([[short], [long]]) == [long]


def test_ties_keep_the_first_group() -> None:
    first = span(0, 4, EntityLabel.CPF)
    second = span(0, 4, EntityLabel.RG)
    assert merge_spans([[first], [second]]) == [first]


def test_overlaps_redacted_once() -> None:
    text = "abcdefghij"
    overlapping = [span(0, 6, EntityLabel.PER, "abcdef"), span(4, 10, EntityLabel.RG, "efghij")]
    assert redact(text, overlapping) == "[PER]ghij"
    apart = [span(0, 4, EntityLabel.CPF, "abcd"), span(6, 8, EntityLabel.RG, "gh")]
    assert redact(text, apart) == "[CPF]ef[RG]ij"


def test_merge_is_pure() -> None:
    groups = [
        [span(0, 6, EntityLabel.PER), span(8, 9, EntityLabel.RG)],
        [span(4, 10, EntityLabel.RG)],
    ]
    before = [[s.model_copy() for s in g] for g in groups]
    merged = merge_spans(groups)
    assert groups == before
    assert [s.start for s in merged] == sorted(s.start for s in merged)
