"""LLM realizer: the model writes a draft with placeholders; `fill()` labels it like a template.

The model only sees non-identifying case facts and slot names, so every identifier in the report
enters through `fill()` and gets an exact span. Drafts get `template_id = "llm:<backend>/<model>"`.
"""

import hashlib
import json
import re
from functools import cache
from importlib.resources import files
from string import Template
from typing import Any

from necrohub.contracts.gold import CaseFacts, Draft
from necrohub.generate.fill import SLOTS
from necrohub.llm.base import ChatLLM, ChatMessage

_FENCE = re.compile(r"^```[a-z]*[ \t]*\n|\n?```$", re.IGNORECASE)


@cache
def load_prompt() -> str:
    """The system prompt shipped as package data, with the slot names filled in."""
    raw = files("necrohub.generate").joinpath("prompts/llm_report.txt").read_text("utf-8")
    slots = "\n".join(f"   {{{{{name}}}}}: {label}" for name, (label, _) in SLOTS.items())
    return Template(raw).substitute(slots=slots)


def case_prompt_facts(case: CaseFacts) -> dict[str, Any]:
    """The case facts without any value a slot carries (names, numbers, places, dates)."""
    facts = case.model_dump(
        mode="json",
        exclude={
            "case_id": True,
            "assistant_name": True,
            "institute": {"city", "uf", "phone"},
            "report": {"number", "exam_date", "police_report_number"},
            "examiner": {"name", "crm"},
            "decedent": {"name", "birthplace", "address", "rg", "cpf", "mother_name"},
            "circumstances": {"estimated_death_at"},
        },
    )
    facts["circumstances"]["estimated_death_time"] = (
        f"{case.circumstances.estimated_death_at:%H:%M}"
    )
    return facts


def llm_seed(seed: int, case_id: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{case_id}:llm".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


class LlmRealizer:
    name = "llm"

    def __init__(self, llm: ChatLLM, prompt: str, temperature: float = 0.8) -> None:
        self.llm, self.prompt, self.temperature = llm, prompt, temperature
        self.template_id = f"llm:{llm.name}"

    def messages(self, case: CaseFacts) -> list[ChatMessage]:
        facts = json.dumps(case_prompt_facts(case), ensure_ascii=False, indent=1)
        return [
            ChatMessage(role="system", content=self.prompt),
            ChatMessage(role="user", content=facts),
        ]

    def realize(self, case: CaseFacts, seed: int) -> Draft:
        """Raises LlmError when the backend fails; the stage records it as a reject."""
        out = self.llm.complete(
            self.messages(case), temperature=self.temperature, seed=llm_seed(seed, case.case_id)
        )
        body = _FENCE.sub("", out.replace("\r\n", "\n").strip()).strip("\n") + "\n"
        return Draft(template_id=self.template_id, body=body)
