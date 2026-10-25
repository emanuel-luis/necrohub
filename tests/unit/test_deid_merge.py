from necrohub.contracts.spans import EntityLabel, PredictedSpan
from necrohub.preprocess.deid.base import merge_spans, redact
from necrohub.preprocess.deid.hf_ner import NerEntity, merge_subwords


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


def _ent(group: str, word: str, start: int, end: int, score: float = 0.9) -> NerEntity:
    return NerEntity(entity_group=group, word=word, start=start, end=end, score=score)


def test_merge_subwords() -> None:
    text = "Corpo de Joãozinho Silva em São Luís"
    entities = [
        _ent("PESSOA", "##zinho", 12, 18, 0.95),  # unsorted on purpose
        _ent("PESSOA", "Jo", 9, 12, 0.8),  # first piece cut mid-word
        _ent("PESSOA", "##ilva", 20, 24),  # orphan piece: widened to the whole word
        _ent("LOCAL", " São Luís", 27, 36),  # leading space is trimmed
    ]
    before = [dict(e) for e in entities]
    out = merge_subwords(entities, text)
    assert entities == before  # input untouched
    assert [(e["start"], e["end"], e["word"]) for e in out] == [
        (9, 18, "Joãozinho"),
        (19, 24, "Silva"),
        (28, 36, "São Luís"),
    ]
    assert out[0]["score"] == 0.95
    assert merge_subwords([], text) == []
