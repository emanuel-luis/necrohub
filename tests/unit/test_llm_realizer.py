import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from necrohub.contracts.gold import CaseFacts, Realized, Reject
from necrohub.generate import stage
from necrohub.generate.fill import SLOTS
from necrohub.generate.llm_realizer import LlmRealizer, load_prompt
from necrohub.generate.realizers import TemplateRealizer
from necrohub.generate.sampler import sample_case
from necrohub.ids import cpf_check_digits
from necrohub.llm.base import ChatMessage, LlmError
from necrohub.params import GenerateParams, LlmParams, Params
from necrohub.preprocess.sections import split_sections
from necrohub.runs import StageFailed, sha256_tree

SEED = 7
CASES = [sample_case(SEED, i) for i in range(5)]

# reply(case, call number) -> the model's answer
Reply = Callable[[CaseFacts, int], str]


def good(case: CaseFacts, _: int = 0) -> str:
    """A well-formed draft: the template realizer's body stands in for a cooperative model."""
    return TemplateRealizer().realize(case, SEED).body


class FakeLLM:
    """Scripted ChatLLM: finds the case from the prompt and answers with reply(case, call)."""

    name = "fake/model"

    def __init__(self, reply: Reply) -> None:
        self.reply = reply
        self.seeds: list[int | None] = []
        self.by_prompt = {LlmRealizer(self, "").messages(c)[-1].content: c for c in CASES}

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
    ) -> str:
        self.seeds.append(seed)
        return self.reply(self.by_prompt[messages[-1].content], len(self.seeds))


def _realize(reply: Reply, case: CaseFacts = CASES[0], attempts: int = 1) -> stage.Outcome:
    llm = FakeLLM(reply)
    return stage.realize_case(LlmRealizer(llm, load_prompt()), case, SEED, 0, attempts)


def _invented_cpf() -> str:
    base = "123456780"[::-1]  # built at runtime: a checksum-valid CPF never appears as a literal
    return f"{base[:3]}.{base[3:6]}.{base[6:]}-{cpf_check_digits(base)}"


def test_good_draft_gives_exact_gold() -> None:
    out = _realize(good)
    assert isinstance(out.result, Realized)
    assert out.template_id == "llm:fake/model" and out.attempts == 1
    assert out.result.sections == split_sections(out.result.text)
    for span in out.result.entities:
        assert out.result.text[span.start : span.end] == span.text


def test_prompt_lists_every_slot_and_facts_hold_no_slot_value() -> None:
    prompt = load_prompt()
    assert all(f"{{{{{name}}}}}" in prompt for name in SLOTS)
    for case in CASES:
        user = LlmRealizer(FakeLLM(good), prompt).messages(case)[-1].content
        json.loads(user)
        for name, (_, get) in SLOTS.items():
            assert get(case) not in user, name


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda b: b.replace("{{CRM}}", "000"), "missing_required"),
        (lambda b: b + "{{NOPE}}\n", "unknown_slot"),
        (lambda b: b + "[[INJURY|ferida\n", "malformed_marker"),
        (lambda b: b + "texto <<SEC:CONCLUSION>>\n", "section_not_at_line_start"),
        (lambda b: b + "fim }}\n", "malformed_marker"),
        (lambda b: b + f"CPF {_invented_cpf()}\n", "invented_identifier"),
        (lambda b: b + "seta →\n", "not_latin1"),
    ],
)
def test_bad_drafts_are_rejected(mutate: Callable[[str], str], reason: str) -> None:
    out = _realize(lambda c, _: mutate(good(c)))
    assert isinstance(out.result, Reject)
    assert out.result.reason == reason
    assert _invented_cpf() not in out.result.detail  # rejects never carry the value


def test_typographic_characters_are_transliterated_and_counted() -> None:
    out = _realize(lambda c, _: good(c) + "Nota — “vide”…\n")
    assert isinstance(out.result, Realized)
    assert out.transliterated == 4
    assert 'Nota - "vide"...\n' in out.result.text


def test_backend_failure_is_a_reject() -> None:
    def down(case: CaseFacts, _: int) -> str:
        raise LlmError("fake: connection refused")

    out = _realize(down)
    assert isinstance(out.result, Reject) and out.result.reason == "llm_error"


def test_retries_use_new_seeds_and_stop_at_max_attempts() -> None:
    llm = FakeLLM(lambda c, _: good(c) + "{{NOPE}}\n")
    out = stage.realize_case(LlmRealizer(llm, load_prompt()), CASES[0], SEED, 0, 3)
    assert isinstance(out.result, Reject) and out.attempts == 3
    assert len(llm.seeds) == 3 and len(set(llm.seeds)) == 3

    out = _realize(lambda c, n: good(c) if n == 2 else "nada\n", attempts=3)
    assert isinstance(out.result, Realized) and out.attempts == 2


def _params(**llm: Any) -> Params:
    return Params(
        seed=SEED,
        generate=GenerateParams(
            n_cases=len(CASES),
            realizer="llm",
            holdout_templates=[],
            max_reject_fraction=0.2,
            pdf_fraction=0.5,
            llm=LlmParams(**llm),
        ),
    )


def _run(tmp: Path, params: Params, reply: Reply, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(stage, "get_llm", lambda backend, model: FakeLLM(reply))
    out = tmp / "raw"
    stage.run_generate(params, out, runs_dir=tmp / "runs", metrics_dir=tmp / "metrics")
    return out


def test_llm_stage_writes_gold_and_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _run(tmp_path, _params(max_attempts=1), good, monkeypatch)
    before = sha256_tree(out)
    golds = sorted((out / "gold").glob("*.json"))
    assert len(golds) == len(CASES)
    assert {json.loads(g.read_text("utf-8"))["realizer"] for g in golds} == {"llm"}
    _run(tmp_path, _params(max_attempts=1), good, monkeypatch)
    assert sha256_tree(out) == before


def test_reject_budget_fails_the_stage_and_keeps_the_previous_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _run(tmp_path, _params(max_attempts=1), good, monkeypatch)
    before = sha256_tree(out)

    def bad_for_two(case: CaseFacts, _: int) -> str:
        return good(case) + ("{{NOPE}}\n" if case.case_id in ("c7-00000", "c7-00001") else "")

    with pytest.raises(StageFailed, match="reject_fraction_within_budget"):
        _run(tmp_path, _params(max_attempts=2), bad_for_two, monkeypatch)
    assert sha256_tree(out) == before
    manifests = [
        json.loads(m.read_text("utf-8")) for m in (tmp_path / "runs" / "generate").glob("*.json")
    ]
    failed = [m for m in manifests if m["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["counts"]["rejected"] == 2 and failed[0]["outputs"] == []
