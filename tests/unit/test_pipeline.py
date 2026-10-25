from necrohub.contracts.documents import ProcessedText
from necrohub.preprocess.pipeline import process


def test_process_returns_a_consistent_record(report: tuple[str, str]) -> None:
    _, text = report
    result = process(text.replace("\n", "\r\n"))
    assert result.text == text
    assert result.normalize_steps[0] == "fix_text" and result.deidentifiers == ["regex"]
    assert ProcessedText.model_validate_json(result.model_dump_json()) == result
    assert all(e.text not in result.redacted_text for e in result.entities)
