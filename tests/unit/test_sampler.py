from datetime import datetime

from necrohub.generate.sampler import sample_case
from necrohub.generate.vocab import SCENARIOS, SPECS, variant_of
from necrohub.ids import fake_values, is_valid_cpf


def test_same_seed_and_index_give_the_same_case() -> None:
    assert sample_case(7, 3) == sample_case(7, 3)
    assert sample_case(7, 3).model_dump_json() != sample_case(8, 3).model_dump_json()


def test_case_id_and_round_robin_scenarios() -> None:
    cases = [sample_case(7, i) for i in range(len(SCENARIOS))]
    assert [c.case_id for c in cases[:2]] == ["c7-00000", "c7-00001"]
    assert {c.scenario for c in cases} == set(SCENARIOS)


def test_identifiers_are_fake_and_text_is_latin1() -> None:
    for i in range(40):
        case = sample_case(11, i)
        assert not is_valid_cpf(case.decedent.cpf)
        assert case.decedent.rg in fake_values("rg")
        assert case.examiner.crm in fake_values("crm")
        assert case.institute.phone in fake_values("phone")
        case.model_dump_json().encode("latin-1")


def test_body_measurements_are_plausible_adult_ones() -> None:
    for i in range(1000):
        case = sample_case(13, i)
        height_m = case.external.height_cm / 100
        assert 18.0 <= case.external.weight_kg / height_m**2 <= 32.0, case.case_id
        assert case.decedent.age >= 16, case.case_id


def test_death_precedes_the_exam_within_the_stated_window() -> None:
    for i in range(20):
        case = sample_case(5, i)
        lo, hi = case.conclusion.time_since_death_h
        start = case.report.exam_date.isoformat() + "T" + case.report.start_time.replace("h", ":")
        gap = datetime.fromisoformat(start) - case.circumstances.estimated_death_at
        assert lo * 3600 <= gap.total_seconds() <= hi * 3600


def test_injuries_places_and_manner_agree_with_the_variant() -> None:
    seen: set[tuple[str, str]] = set()
    for i in range(400):
        case = sample_case(3, i)
        spec = SPECS[case.scenario]
        variant = variant_of(case.scenario, case.conclusion.cause_of_death)
        seen.add((case.scenario, variant.cause))
        allowed = {
            (s.kind, region)
            for s in (*variant.injuries, *spec.injuries, *spec.optional_injuries)
            for region in s.regions
        }
        for injury in case.external.injuries:
            assert (injury.kind, injury.region) in allowed, (case.case_id, injury)
        kinds = [x.kind for x in case.external.injuries]
        assert all(s.kind in kinds for s in (*variant.injuries, *spec.injuries))
        assert case.circumstances.place_found in (variant.places or spec.places)
        assert case.conclusion.manner_of_death in (variant.manners or spec.manners)
        abnormal = {o.organ: o.finding for o in case.internal if o.abnormal}
        assert abnormal == variant.findings
    assert seen == {(name, v.cause) for name, s in SPECS.items() for v in s.variants}
