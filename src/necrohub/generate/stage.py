"""The generate stage: params -> raw layer (reports, gold records, rejects) plus metrics."""

from collections import Counter
from pathlib import Path

from necrohub.contracts.gold import GoldFiles, GoldRecord, Reject, text_sha256
from necrohub.generate.checks import check_raw
from necrohub.generate.fill import fill
from necrohub.generate.realizers import TemplateRealizer, TextRealizer, load_families
from necrohub.generate.sampler import sample_case
from necrohub.params import Params
from necrohub.runs import (
    Counts,
    FileRef,
    RunManifest,
    atomic_dir,
    canonical_json,
    sha256_tree,
    stage_run,
    write_jsonl,
    write_text_atomic,
)


def run_generate(
    params: Params,
    out: Path,
    *,
    runs_dir: Path = Path("runs"),
    metrics_dir: Path = Path("metrics"),
) -> RunManifest:
    """Write <out>/reports/<case_id>.txt, <out>/gold/<case_id>.json and <out>/rejects.jsonl.

    Raises StageFailed (previous out kept, manifest marked failed) when a boundary check fails.
    """
    g = params.generate
    realizer: TextRealizer = TemplateRealizer()
    # the template realizer must reject nothing; P2b's LLM realizer gets the configured budget
    budget = 0.0 if g.realizer == "template" else g.max_reject_fraction
    stage_params = {"seed": params.seed, "generate": g.model_dump()}
    with stage_run("generate", stage_params, runs_dir) as run, atomic_dir(out, run.run_id) as tmp:
        golds: list[GoldRecord] = []
        rejects: list[Reject] = []
        for i in range(g.n_cases):
            case = sample_case(params.seed, i)
            draft = realizer.realize(case, params.seed)
            result = fill(draft, case)
            if isinstance(result, Reject):
                rejects.append(result)
                continue
            txt = f"reports/{case.case_id}.txt"
            (tmp / "reports").mkdir(exist_ok=True)
            (tmp / txt).write_text(result.text, encoding="utf-8", newline="\n")
            gold = GoldRecord(
                case=case,
                realizer=realizer.name,
                template_id=draft.template_id,
                text_sha256=text_sha256(result.text),
                entities=result.entities,
                sections=result.sections,
                files=GoldFiles(txt=txt),
            )
            (tmp / "gold").mkdir(exist_ok=True)
            (tmp / "gold" / f"{case.case_id}.json").write_text(
                gold.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
            )
            golds.append(gold)
        write_jsonl(tmp / "rejects.jsonl", rejects)

        for name, passed, detail in check_raw(tmp, g.n_cases, budget):
            run.check(name, passed, detail)
        unknown = sorted(set(g.holdout_templates) - set(load_families()))
        run.check("holdout_templates_known", not unknown, ", ".join(unknown))

        tree = sha256_tree(tmp)
        run.counts = Counts(read=g.n_cases, written=len(golds), rejected=len(rejects))
        run.rejects_path = (out / "rejects.jsonl").as_posix()
        run.outputs = [FileRef(path=out.as_posix(), sha256=tree, records=len(golds))]
        run.raise_if_failed()

    metrics = {
        "n_cases": g.n_cases,
        "written": len(golds),
        "rejected": len(rejects),
        "raw_sha256": tree,
        "by_scenario": dict(Counter(x.case.scenario for x in golds)),
        "by_template": dict(Counter(x.template_id for x in golds)),
        "entities_by_label": dict(Counter(str(s.label) for x in golds for s in x.entities)),
        "reject_reasons": dict(Counter(r.reason for r in rejects)),
    }
    write_text_atomic(metrics_dir / "generate.json", canonical_json(metrics))
    return run
