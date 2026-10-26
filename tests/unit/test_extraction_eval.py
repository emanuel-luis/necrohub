import json
from pathlib import Path

from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.contracts.datasets import CaseExtraction, ExtractionRecord
from necrohub.contracts.gold import Injury, OrganFinding
from necrohub.evaluate.extraction import parse_extraction, score_extraction
from necrohub.runs import write_jsonl

runner = CliRunner()


def _gold(cause: str = "Choque hemorrágico") -> CaseExtraction:
    return CaseExtraction(
        scenario="traffic",
        cause_of_death=cause,
        mechanism="Hemorragia",
        manner_of_death="Violenta",
        time_since_death_h=(6, 12),
        injuries=[
            Injury(kind="Escoriação", region="Tórax", side=None, size_cm=2.5, features=["a", "b"])
        ],
        organ_findings=[
            OrganFinding(organ="Fígado", finding="Laceração", abnormal=True),
            OrganFinding(organ="Rim", finding="Sem alterações", abnormal=False),
        ],
    )


def test_parse_accepts_bare_fenced_and_prose_wrapped_json() -> None:
    raw = _gold().model_dump_json()
    assert parse_extraction(raw) == _gold()
    assert parse_extraction(f"```json\n{raw}\n```") == _gold()
    assert parse_extraction(f"Segue o resultado: {raw} Espero que ajude. {{}}") == _gold()
    assert parse_extraction('{"scenario": "traffic"}') is None
    assert parse_extraction("sem json") is None
    assert parse_extraction(raw[:-5]) is None


def test_perfect_prediction() -> None:
    s = score_extraction([_gold()], [_gold()])
    assert s["valid_json_rate"] == s["all_fields_exact"] == 1.0
    assert all(v == 1.0 for v in s["field_accuracy"].values())
    assert s["injuries"]["f1"] == s["organ_findings"]["f1"] == 1.0


def test_normalisation_and_one_wrong_field() -> None:
    pred = _gold(" CHOQUE   hemorrágico").model_copy(update={"mechanism": "Outra"})
    s = score_extraction([_gold()], [pred])
    assert s["field_accuracy"]["cause_of_death"] == 1.0
    assert s["field_accuracy"]["mechanism"] == 0.0
    assert s["all_fields_exact"] == 0.0 and s["valid_json_rate"] == 1.0


def test_invalid_prediction_is_wrong_everywhere() -> None:
    s = score_extraction([_gold(), _gold()], [_gold(), None])
    assert s["valid_json_rate"] == 0.5
    assert s["field_accuracy"]["scenario"] == 0.5
    assert s["injuries"]["fn"] == 1 and s["organ_findings"]["fn"] == 2
    assert s["all_fields_exact"] == 0.5


def test_set_scores_collapse_duplicates() -> None:
    g = _gold()
    dup = g.model_copy(update={"injuries": g.injuries * 3, "organ_findings": g.organ_findings[:1]})
    s = score_extraction([g], [dup])
    assert s["injuries"]["tp"] == 1 and s["injuries"]["fp"] == 0
    assert s["organ_findings"]["tp"] == 1 and s["organ_findings"]["fn"] == 1
    assert s["organ_findings"]["recall"] == 0.5


def test_eval_extraction_cli(tmp_path: Path) -> None:
    recs = [
        ExtractionRecord(id=f"d{i}:extraction", doc_id=f"d{i}", input_text="x", target=_gold())
        for i in range(3)
    ]
    write_jsonl(tmp_path / "extraction" / "test.jsonl", recs)
    preds = tmp_path / "preds.jsonl"
    lines = [
        {"id": "d0:extraction", "text": _gold().model_dump_json()},
        {"id": "d1:extraction", "text": "not json"},
    ]
    preds.write_text("\n".join(json.dumps(x) for x in lines) + "\n", "utf-8")
    out = tmp_path / "results.json"
    args = ["eval-extraction", "--data", str(tmp_path), "--predictions", str(preds)]
    result = runner.invoke(app, [*args, "--out", str(out)])
    assert result.exit_code == 0, result.output
    got = json.loads(out.read_text("utf-8"))
    assert got["n"] == 3 and got["missing_predictions"] == 1
    assert got["valid_json_rate"] == round(1 / 3, 4)
    bad = runner.invoke(app, ["eval-extraction", "--predictions", str(preds), "--split", "x"])
    assert bad.exit_code == 2
