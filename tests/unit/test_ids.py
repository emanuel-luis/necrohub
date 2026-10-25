import random
import re

from hypothesis import assume, given
from hypothesis import strategies as st

from necrohub.ids import (
    cnpj_check_digits,
    cpf_check_digits,
    fake_cpf,
    is_valid_cnpj,
    is_valid_cpf,
    load_pool,
)


@given(st.from_regex(r"[0-9]{9}", fullmatch=True))
def test_cpf_check_digits(base: str) -> None:
    assume(len(set(base)) > 1)
    cpf = base + cpf_check_digits(base)
    assert is_valid_cpf(cpf)
    assert is_valid_cpf(f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}")
    assert not is_valid_cpf(cpf[:10] + str((int(cpf[10]) + 1) % 10))


@given(st.from_regex(r"[0-9]{12}", fullmatch=True))
def test_cnpj_check_digits(base: str) -> None:
    assume(len(set(base)) > 1)
    cnpj = base + cnpj_check_digits(base)
    assert is_valid_cnpj(cnpj)
    assert not is_valid_cnpj(cnpj[:13] + str((int(cnpj[13]) + 1) % 10))


def test_repeated_digits_are_not_valid() -> None:
    assert not is_valid_cpf("1" * 11) and not is_valid_cnpj("0" * 14)


@given(st.integers())
def test_fake_cpf_checksum_invalid(seed: int) -> None:
    cpf = fake_cpf(random.Random(seed))
    assert re.fullmatch(r"\d{3}\.\d{3}\.\d{3}-\d{2}", cpf)
    assert not is_valid_cpf(cpf)


def test_pool_holds_the_fixture_identifiers() -> None:
    pool = load_pool()
    assert {"125623240", "1234567", "7654321"} <= pool.rg
    assert {"12345", "22456"} <= pool.crm
    assert "12345678900" in pool.cpf
    assert "example.org" in pool.email_domains
