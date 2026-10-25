"""Seeded case sampler: (seed, i) -> CaseFacts, byte-identical across runs and platforms."""

import random
from datetime import date, datetime, timedelta
from functools import cache
from typing import Literal

from faker import Faker

from necrohub.contracts.gold import (
    CaseFacts,
    Circumstances,
    Conclusion,
    Decedent,
    Examiner,
    External,
    Injury,
    Institute,
    OrganFinding,
    ReportInfo,
)
from necrohub.generate import vocab
from necrohub.ids import fake_cpf, fake_values

_FIRST_EXAM_DAY = date(2024, 1, 1)
_EXAM_DAYS = 731  # 2024-01-01 .. 2025-12-31; never the wall clock


@cache
def _faker() -> Faker:
    return Faker("pt_BR")


def case_id(seed: int, i: int) -> str:
    return f"c{seed}-{i:05d}"


def case_index(case_id: str) -> int:
    """The index i that case_id(seed, i) was built from."""
    return int(case_id.rsplit("-", 1)[1])


def case_rng(seed: int, i: int) -> random.Random:
    """The case's own generator; str seeds hash with SHA-512, the same on every platform."""
    return random.Random(f"{seed}:{i}")


def _person(fake: Faker, sex: str) -> str:
    first = fake.first_name_male() if sex == "masculino" else fake.first_name_female()
    return f"{first} {fake.last_name()} {fake.last_name()}"


def _hhmm(t: datetime) -> str:
    return f"{t.hour:02d}h{t.minute:02d}"


def _injury(spec: vocab.InjurySpec, rng: random.Random) -> Injury:
    k = rng.randint(1, min(2, len(spec.features)))
    return Injury(
        kind=spec.kind,
        region=rng.choice(spec.regions),
        side=rng.choice(spec.sides),
        size_cm=round(rng.uniform(*spec.size), 1) if spec.size else None,
        features=rng.sample(spec.features, k),
    )


def sample_case(seed: int, i: int) -> CaseFacts:
    rng = case_rng(seed, i)
    fake = _faker()
    fake.seed_instance(f"{seed}:{i}")
    scenario = vocab.SCENARIOS[i % len(vocab.SCENARIOS)]  # round-robin: every scenario appears
    spec = vocab.SPECS[scenario]

    variant = rng.choice(spec.variants)  # first: it constrains places, injuries and manner
    city, uf, ddd = rng.choice(vocab.CITIES)
    phones = [p for p in fake_values("phone") if p.startswith(f"({ddd})")]
    exam_day = _FIRST_EXAM_DAY + timedelta(days=rng.randrange(_EXAM_DAYS))
    start = datetime.combine(exam_day, datetime.min.time()) + timedelta(
        hours=rng.randint(7, 13), minutes=rng.choice((0, 15, 30, 45))
    )
    end = start + timedelta(minutes=rng.randrange(60, 155, 5))
    state = vocab.STATES[rng.choice(spec.states)]
    death_at = start - timedelta(
        minutes=rng.randrange(state.hours[0] * 60, state.hours[1] * 60, 30)
    )

    sex = "masculino" if rng.random() < spec.male_share else "feminino"
    tall = (160, 190) if sex == "masculino" else (150, 175)
    height_cm = rng.randint(*tall)
    # adult BMI 18.5-31.5, so the weight agrees with the height
    weight_kg = round(rng.uniform(18.5, 31.5) * (height_cm / 100) ** 2)
    examiner_sex: Literal["masculino", "feminino"] = rng.choice(("masculino", "feminino"))
    birth_city, birth_uf, _ = rng.choice(vocab.CITIES)
    optional = rng.sample(spec.optional_injuries, rng.randint(0, len(spec.optional_injuries)))
    extra_exams = rng.sample(spec.exams, rng.randint(1, len(spec.exams)))

    return CaseFacts(
        case_id=case_id(seed, i),
        scenario=scenario,
        institute=Institute(
            name=rng.choice(vocab.INSTITUTES), city=city, uf=uf, phone=rng.choice(phones)
        ),
        report=ReportInfo(
            number=f"{exam_day.year}/{rng.randrange(1000, 10000)}",
            exam_date=exam_day,
            start_time=_hhmm(start),
            end_time=_hhmm(end),
            police_report_number=f"{rng.randrange(1000, 100000)}/{exam_day.year}",
        ),
        examiner=Examiner(
            name=_person(fake, examiner_sex),
            sex=examiner_sex,
            crm=rng.choice(fake_values("crm")),
            crm_uf=uf,
        ),
        assistant_name=_person(fake, rng.choice(("masculino", "feminino"))),
        decedent=Decedent(
            name=_person(fake, sex),
            sex=sex,
            age=rng.randint(*spec.ages),
            skin_color=rng.choice(vocab.SKIN_COLORS),
            birthplace=f"{birth_city}/{birth_uf}",
            address=f"{rng.choice(vocab.STREET_TYPES)} {fake.first_name()} {fake.last_name()}, "
            f"nº {rng.randint(1, 2500)}, {rng.choice(vocab.NEIGHBOURHOODS)}",
            rg=rng.choice(fake_values("rg")),
            cpf=fake_cpf(rng),
            mother_name=_person(fake, "feminino"),
        ),
        circumstances=Circumstances(
            place_found=rng.choice(variant.places or spec.places),
            body_position=rng.choice(spec.positions),
            estimated_death_at=death_at,
        ),
        external=External(
            height_cm=height_cm,
            weight_kg=weight_kg,
            rigor=state.rigor,
            livor=state.livor,
            decomposition=state.decomposition,
            injuries=[_injury(s, rng) for s in (*variant.injuries, *spec.injuries, *optional)],
        ),
        internal=[
            OrganFinding(organ=organ, finding=variant.findings[organ], abnormal=True)
            if organ in variant.findings
            else OrganFinding(
                organ=organ, finding=rng.choice(vocab.NORMAL_FINDINGS), abnormal=False
            )
            for organ in vocab.ORGANS
            if organ in variant.findings or organ not in vocab.OPTIONAL_ORGANS
        ],
        complementary_exams=[*variant.exams, *extra_exams],
        conclusion=Conclusion(
            cause_of_death=variant.cause,
            mechanism=variant.mechanism,
            manner_of_death=rng.choice(variant.manners or spec.manners),
            time_since_death_h=state.hours,
        ),
    )
