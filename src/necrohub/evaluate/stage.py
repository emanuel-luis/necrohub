"""The evaluate stage: test records, split by gold template, scored for regex and the encoder."""

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from necrohub.contracts.datasets import NerRecord, SectionRecord
from necrohub.contracts.gold import GoldRecord
from necrohub.contracts.spans import Span
from necrohub.evaluate.metrics import ner_scores, section_scores
from necrohub.params import Params
from necrohub.preprocess.deid.base import merge_spans
from necrohub.preprocess.deid.regex import RegexDeidentifier
from necrohub.runs import (
    Counts,
    FileRef,
    RunManifest,
    canonical_json,
    read_jsonl,
    sha256_tree,
    stage_run,
    write_text_atomic,
)
from necrohub.settings import resolve_device
from necrohub.train.encoder import load_trained, predict_ner, predict_sections

SCOPES = ("random", "heldout_templates")


def _base_model(model_dir: Path) -> str:
    info = json.loads((model_dir / "necrohub_train.json").read_text("utf-8"))
    return str(info["base_model"])


def _by_scope[R](
    records: Sequence[R], doc_id: Callable[[R], str], holdout_of: dict[str, bool]
) -> dict[str, list[int]]:
    """Indices of the records in each scope, by the gold template of their document."""
    out: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        out["heldout_templates" if holdout_of[doc_id(r)] else "random"].append(i)
    return out


def run_evaluate(
    params: Params,
    models_dir: Path,
    data_dir: Path,
    raw_dir: Path,
    out_json: Path,
    *,
    runs_dir: Path = Path("runs"),
) -> RunManifest:
    """Score the encoders (and the regex baseline) on the test split and write `out_json`.

    Test records only. A record's scope is `heldout_templates` when the gold template of its
    case is in generate.holdout_templates, else `random`. The regex de-identifier predicts PII
    only, so it is scored against the gold PII spans; the encoder against every label. An
    empty scope is reported with n_records 0 and no metrics.
    """
    holdout = params.generate.holdout_templates
    t = params.train
    stage_params = {
        "seed": params.seed,
        "holdout_templates": holdout,
        "evaluate": params.evaluate.model_dump(),
        "max_length": t.max_length,
        "stride": t.stride,
    }
    with stage_run("evaluate", stage_params, runs_dir) as run:
        ner_dir, sec_dir = models_dir / "encoder-ner", models_dir / "encoder-sections"
        run.inputs = [
            FileRef(path=data_dir.as_posix(), sha256=sha256_tree(data_dir)),
            FileRef(path=ner_dir.as_posix(), sha256=sha256_tree(ner_dir)),
            FileRef(path=sec_dir.as_posix(), sha256=sha256_tree(sec_dir)),
        ]
        template = {
            f.stem: GoldRecord.model_validate_json(f.read_text("utf-8")).template_id
            for f in sorted((raw_dir / "gold").glob("*.json"))
        }
        ner = read_jsonl(data_dir / "ner" / "test.jsonl", NerRecord)
        sections = read_jsonl(data_dir / "sections" / "test.jsonl", SectionRecord)
        missing = ({r.doc_id for r in ner} | {r.doc_id for r in sections}) - template.keys()
        run.check("test_docs_have_gold", not missing, f"{len(missing)} without gold")
        run.raise_if_failed()
        is_held = {d: tpl in holdout for d, tpl in template.items()}
        device = resolve_device("auto")

        model, tok = load_trained(ner_dir, "ner")
        predicted = predict_ner(model, tok, [r.text for r in ner], t.max_length, t.stride, device)
        regex = RegexDeidentifier()
        regex_pred: list[Sequence[Span]] = [merge_spans([regex.detect(r.text)]) for r in ner]
        ner_out: dict[str, Any] = {}
        for scope, idx in _by_scope(ner, lambda r: r.doc_id, is_held).items():
            gold = [ner[i].spans for i in idx]
            entry: dict[str, Any] = {
                "n_records": len(idx),
                "n_cases": len({ner[i].case_id for i in idx}),
            }
            if "regex" in params.evaluate.baselines:
                pii_gold = [[s for s in g if s.label.is_pii] for g in gold]
                entry["regex"] = ner_scores(pii_gold, [regex_pred[i] for i in idx])
            entry["encoder"] = ner_scores(gold, [predicted[i] for i in idx])
            ner_out[scope] = entry

        model, tok = load_trained(sec_dir, "sections")
        labels = predict_sections(model, tok, [r.text for r in sections], t.max_length, device)
        sec_out: dict[str, Any] = {}
        for scope, idx in _by_scope(sections, lambda r: r.doc_id, is_held).items():
            sec_out[scope] = {
                "n_records": len(idx),
                "encoder": section_scores(
                    [str(sections[i].label) for i in idx], [str(labels[i]) for i in idx]
                ),
            }
        for scope in SCOPES:
            ner_out.setdefault(scope, {"n_records": 0, "n_cases": 0})
            sec_out.setdefault(scope, {"n_records": 0})
        run.counts = Counts(read=len(ner) + len(sections), written=1)

    result = {
        "schema_version": 1,
        "seed": params.seed,
        "datasets_sha256": sha256_tree(data_dir),
        "base_model": {"ner": _base_model(ner_dir), "sections": _base_model(sec_dir)},
        "ner": dict(sorted(ner_out.items())),
        "sections": dict(sorted(sec_out.items())),
    }
    write_text_atomic(out_json, canonical_json(result))
    return run
