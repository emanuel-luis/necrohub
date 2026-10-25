from necrohub.contracts.documents import ProcessedText
from necrohub.contracts.spans import EntityLabel, PredictedSpan
from necrohub.preprocess.pipeline import process


def test_process_returns_a_consistent_record(report: tuple[str, str]) -> None:
    _, text = report
    result = process(text.replace("\n", "\r\n"))
    assert result.text == text
    assert result.normalize_steps[0] == "fix_text" and result.deidentifiers == ["regex"]
    assert ProcessedText.model_validate_json(result.model_dump_json()) == result
    assert all(e.text not in result.redacted_text for e in result.entities)


class _Fixed:
    """A detector that returns fixed spans, to combine overlapping detectors."""

    def __init__(self, name: str, *ranges: tuple[int, int, EntityLabel]) -> None:
        self.name = name
        self._ranges = ranges

    def detect(self, text: str) -> list[PredictedSpan]:
        return [
            PredictedSpan(start=s, end=e, label=lab, text=text[s:e], score=1.0, source="regex")
            for s, e, lab in self._ranges
        ]


def test_partial_overlap_across_detectors_is_fully_redacted() -> None:
    text = "abcdefghij tail"
    first = _Fixed("a", (0, 6, EntityLabel.PER))
    second = _Fixed("b", (4, 10, EntityLabel.RG))
    result = process(text, deidentifiers=(first, second))
    assert [e.label for e in result.entities] == [EntityLabel.PER]
    assert result.redacted_text == "[PER] tail"
