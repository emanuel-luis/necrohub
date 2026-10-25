"""The generate stage: params -> raw layer (reports, gold records, rejects) plus metrics."""

import hashlib
from collections import Counter
from functools import partial
from pathlib import Path
from typing import NamedTuple

from necrohub.contracts.gold import (
    CaseFacts,
    Draft,
    GoldFiles,
    GoldRecord,
    Realized,
    Reject,
    text_sha256,
)
from necrohub.generate.checks import check_raw, unlabeled_identifiers
from necrohub.generate.fill import fill
from necrohub.generate.llm_realizer import LlmRealizer, load_prompt
from necrohub.generate.realizers import TemplateRealizer, TextRealizer
from necrohub.generate.render_pdf import render_pdf, transliterate, wants_pdf
from necrohub.generate.sampler import sample_case
from necrohub.llm.base import LlmError, get_llm
from necrohub.params import GenerateParams, Params
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


class Outcome(NamedTuple):
    template_id: str
    result: Realized | Reject
    attempts: int
    transliterated: int


def attempt_seed(seed: int, i: int, attempt: int) -> int:
    """The run seed for the first attempt, a seed derived from (seed, i, attempt) for retries."""
    if attempt == 0:
        return seed
    digest = hashlib.blake2b(f"{seed}:{i}:{attempt}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _realize_once(realizer: TextRealizer, case: CaseFacts, seed: int) -> Outcome:
    try:
        draft = realizer.realize(case, seed)
    except LlmError as exc:
        reject = Reject(
            case_id=case.case_id, template_id=realizer.name, reason="llm_error", detail=str(exc)
        )
        return Outcome(realizer.name, reject, 1, 0)
    # an LLM's typographic dashes and quotes become latin-1 before fill, so spans stay exact
    body, n = transliterate(draft.body)
    draft = Draft(template_id=draft.template_id, body=body)
    result = fill(draft, case)
    if isinstance(result, Realized):
        rejected = partial(Reject, case_id=case.case_id, template_id=draft.template_id)
        if leaks := unlabeled_identifiers(result.text, result.entities):
            # kinds and offsets only, never the value
            result = rejected(reason="invented_identifier", detail=", ".join(leaks[:5]))
        elif bad := [c for c in result.text if ord(c) > 0xFF]:
            result = rejected(reason="not_latin1", detail=f"{len(bad)} char(s), first {bad[0]!r}")
    return Outcome(draft.template_id, result, 1, n)


def realize_case(
    realizer: TextRealizer, case: CaseFacts, seed: int, i: int, max_attempts: int = 1
) -> Outcome:
    """Realize, fill and check one case, retrying a reject with a new seed up to max_attempts.

    Returns the first accepted result, or the last reject; never raises for a bad draft.
    """
    transliterated = 0
    for attempt in range(max_attempts):
        out = _realize_once(realizer, case, attempt_seed(seed, i, attempt))
        transliterated += out.transliterated
        if isinstance(out.result, Realized):
            break
    return out._replace(attempts=attempt + 1, transliterated=transliterated)


def build_realizer(g: GenerateParams) -> TextRealizer:
    if g.realizer == "llm":
        return LlmRealizer(
            get_llm(g.llm.backend, g.llm.model), load_prompt(), temperature=g.llm.temperature
        )
    return TemplateRealizer(
        holdout_templates=g.holdout_templates, holdout_fraction=g.holdout_fraction
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
    # the template realizer must reject nothing; P2b's LLM realizer gets the configured budget
    budget = 0.0 if g.realizer == "template" else g.max_reject_fraction
    stage_params = {"seed": params.seed, "generate": g.model_dump()}
    with stage_run("generate", stage_params, runs_dir) as run, atomic_dir(out, run.run_id) as tmp:
        # inside the run: a bad holdout setting or a missing LLM key fails the run with a manifest
        realizer = build_realizer(g)
        attempts = 1 if g.realizer == "template" else g.llm.max_attempts
        golds: list[GoldRecord] = []
        rejects: list[Reject] = []
        n_attempts = n_transliterated = n_pdf_chars = 0
        for i in range(g.n_cases):
            case = sample_case(params.seed, i)
            outcome = realize_case(realizer, case, params.seed, i, attempts)
            n_attempts += outcome.attempts
            n_transliterated += outcome.transliterated
            result = outcome.result
            if isinstance(result, Reject):
                rejects.append(result)
                continue
            txt = f"reports/{case.case_id}.txt"
            (tmp / "reports").mkdir(exist_ok=True)
            (tmp / txt).write_text(result.text, encoding="utf-8", newline="\n")
            pdf = None
            if wants_pdf(params.seed, case.case_id, g.pdf_fraction):
                pdf = f"reports/{case.case_id}.pdf"
                data, n_replaced = render_pdf(result.text)
                (tmp / pdf).write_bytes(data)
                n_pdf_chars += n_replaced
            gold = GoldRecord(
                case=case,
                realizer=realizer.name,
                template_id=outcome.template_id,
                text_sha256=text_sha256(result.text),
                entities=result.entities,
                sections=result.sections,
                files=GoldFiles(txt=txt, pdf=pdf),
            )
            (tmp / "gold").mkdir(exist_ok=True)
            (tmp / "gold" / f"{case.case_id}.json").write_text(
                gold.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
            )
            golds.append(gold)
        write_jsonl(tmp / "rejects.jsonl", rejects)

        for name, passed, detail in check_raw(tmp, g.n_cases, budget):
            run.check(name, passed, detail)

        tree = sha256_tree(tmp)
        run.counts = Counts(read=g.n_cases, written=len(golds), rejected=len(rejects))
        run.rejects_path = (out / "rejects.jsonl").as_posix()
        run.raise_if_failed()
        # only a run that passed its checks claims an output
        run.outputs = [FileRef(path=out.as_posix(), sha256=tree, records=len(golds))]

    metrics = {
        "n_cases": g.n_cases,
        "written": len(golds),
        "rejected": len(rejects),
        "raw_sha256": tree,
        "by_scenario": dict(Counter(x.case.scenario for x in golds)),
        "by_template": dict(Counter(x.template_id for x in golds)),
        "entities_by_label": dict(Counter(str(s.label) for x in golds for s in x.entities)),
        "reject_reasons": dict(Counter(r.reason for r in rejects)),
        "attempts": n_attempts,
        "transliterated_chars": n_transliterated,
        "pdfs": sum(1 for x in golds if x.files.pdf),
        "pdf_replaced_chars": n_pdf_chars,
    }
    write_text_atomic(metrics_dir / "generate.json", canonical_json(metrics))
    return run
