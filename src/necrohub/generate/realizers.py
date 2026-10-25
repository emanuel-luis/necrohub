"""Realizers turn CaseFacts into a Draft; `fill()` then turns the draft into text and spans."""

import random
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from string import Template
from typing import Protocol

from necrohub.contracts.gold import CaseFacts, Draft
from necrohub.generate.vocab import SPECS, SUBSTANCES


class TextRealizer(Protocol):
    name: str

    def realize(self, case: CaseFacts, seed: int) -> Draft: ...


@dataclass(frozen=True)
class Family:
    """One whole-report template: a body plus phrase variants for list items."""

    body: str
    injury: tuple[str, ...]
    measure: tuple[str, ...]
    no_injuries: tuple[str, ...]
    organ: tuple[str, ...]
    exam: tuple[str, ...]


@cache
def load_families() -> dict[str, Family]:
    """Every templates/<id>.toml shipped with the package, by id."""
    out: dict[str, Family] = {}
    for f in sorted(
        files("necrohub.generate").joinpath("templates").iterdir(), key=lambda f: f.name
    ):
        if f.name.endswith(".toml"):
            raw = tomllib.loads(f.read_text("utf-8"))
            out[f.name.removesuffix(".toml")] = Family(
                body=raw["body"],
                **{k: tuple(raw[k]) for k in ("injury", "measure", "no_injuries", "organ", "exam")},
            )
    return out


_SUBSTANCE = re.compile(rf"\b({'|'.join(SUBSTANCES)})\b")


def _mark_substances(text: str) -> str:
    return _SUBSTANCE.sub(r"[[SUBSTANCE|\1]]", text)


def _num(x: float, digits: int) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " e " + items[-1]


class TemplateRealizer:
    """Deterministic realizer: picks a family and phrase variants with a (seed, case) rng."""

    name = "template"

    def __init__(self, families: Mapping[str, Family] | None = None) -> None:
        self.families = dict(families or load_families())

    def realize(self, case: CaseFacts, seed: int) -> Draft:
        rng = random.Random(f"{seed}:{case.case_id}:template")
        family_id = rng.choice(sorted(self.families))
        return Draft(template_id=family_id, body=self.render(case, family_id, rng))

    def render(self, case: CaseFacts, family_id: str, rng: random.Random) -> str:
        fam = self.families[family_id]
        d, ext, con = case.decedent, case.external, case.conclusion

        injuries = []
        for inj in ext.injuries:
            kind = f"[[INJURY|{inj.kind}]]"
            region = inj.region + (f", à {inj.side}" if inj.side else "")
            measure = (
                Template(rng.choice(fam.measure)).substitute(size=_num(inj.size_cm, 1))
                if inj.size_cm is not None
                else ""
            )
            injuries.append(
                Template(rng.choice(fam.injury)).substitute(
                    kind=kind,
                    Kind=f"[[INJURY|{_cap(inj.kind)}]]",
                    region=region,
                    measure=measure,
                    features=_join(inj.features),
                )
            )
        if not injuries:
            injuries.append(rng.choice(fam.no_injuries))

        internal = [
            Template(rng.choice(fam.organ)).substitute(
                organ=f"[[ORGAN|{o.organ}]]",
                finding=f"[[FINDING|{o.finding}]]" if o.abnormal else o.finding,
            )
            for o in case.internal
        ]
        exams = [
            Template(rng.choice(fam.exam)).substitute(
                exam=_mark_substances(e), Exam=_mark_substances(_cap(e))
            )
            for e in case.complementary_exams
        ]
        circ = case.circumstances
        substance = next((s for s in SUBSTANCES if s in con.cause_of_death.lower()), "")
        history = Template(rng.choice(SPECS[case.scenario].histories)).substitute(
            place=circ.place_found,
            position=circ.body_position,
            time=f"{circ.estimated_death_at:%H}h{circ.estimated_death_at:%M}",
            substance=substance,
        )
        body = Template(fam.body).substitute(
            institute=case.institute.name,
            start=case.report.start_time,
            end=case.report.end_time,
            crm_uf=case.examiner.crm_uf,
            sex=d.sex,
            age=d.age,
            skin=d.skin_color,
            child_of="filho de" if d.sex == "masculino" else "filha de",
            history=_mark_substances(history),
            height=_num(ext.height_cm / 100, 2),
            weight=ext.weight_kg,
            rigor=ext.rigor,
            livor=ext.livor,
            decomp=ext.decomposition,
            injuries="\n".join(injuries),
            internal="\n".join(internal),
            exams="\n".join(exams),
            cause=f"[[CAUSE_OF_DEATH|{con.cause_of_death}]]",
            mechanism=con.mechanism,
            manner=f"[[MANNER_OF_DEATH|{con.manner_of_death}]]",
            h_lo=con.time_since_death_h[0],
            h_hi=con.time_since_death_h[1],
        )
        return body.lstrip("\n").rstrip("\n") + "\n"
