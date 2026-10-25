"""Brazilian identifiers: check digits, value patterns, the fake pool and the leak check."""

import random
import re
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path

# --- check digits ---------------------------------------------------------------------------

_CPF_WEIGHTS = tuple(range(11, 1, -1))
_CNPJ_WEIGHTS = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)


def _check(digits: str, weights: tuple[int, ...]) -> str:
    # the shorter base uses the tail of the weights: 10..2 for CPF, 5..2,9..2 for CNPJ
    rest = sum(int(d) * w for d, w in zip(digits, weights[-len(digits) :], strict=True)) % 11
    return str(0 if rest < 2 else 11 - rest)


def cpf_check_digits(base: str) -> str:
    """The two check digits for the first nine digits of a CPF."""
    first = _check(base, _CPF_WEIGHTS)
    return first + _check(base + first, _CPF_WEIGHTS)


def cnpj_check_digits(base: str) -> str:
    """The two check digits for the first twelve digits of a CNPJ."""
    first = _check(base, _CNPJ_WEIGHTS)
    return first + _check(base + first, _CNPJ_WEIGHTS)


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def is_valid_cpf(value: str) -> bool:
    d = _digits(value)
    return len(d) == 11 and len(set(d)) > 1 and d[9:] == cpf_check_digits(d[:9])


def is_valid_cnpj(value: str) -> bool:
    d = _digits(value)
    return len(d) == 14 and len(set(d)) > 1 and d[12:] == cnpj_check_digits(d[:12])


def fake_cpf(rng: random.Random) -> str:
    """A well-formed CPF whose last check digit is wrong, so it can never be a real person's."""
    base = "".join(str(rng.randrange(10)) for _ in range(9))
    digits = cpf_check_digits(base)
    wrong = (int(digits[1]) + rng.randrange(1, 10)) % 10
    return f"{base[:3]}.{base[3:6]}.{base[6:]}-{digits[0]}{wrong}"


# --- value patterns -------------------------------------------------------------------------
# Each pattern has a `value` group: only the value is redacted, the "CPF nº" label stays.

_CTX = r"(?:\s*(?:n[ºo°]\.?|:))?\s*"


def _compile(*patterns: str, flags: int = 0) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, flags) for p in patterns)


IDENTIFIER_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "CPF": _compile(
        r"(?<![\d.])(?P<value>\d{3}\.\d{3}\.\d{3}-\d{2})(?!\d)",
        rf"\bCPF\b{_CTX}(?P<value>\d{{11}})(?!\d)",
    ),
    "CNPJ": _compile(
        r"(?<![\d.])(?P<value>\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})(?!\d)",
        rf"\bCNPJ\b{_CTX}(?P<value>\d{{14}})(?!\d)",
    ),
    "RG": _compile(rf"\bRG\b{_CTX}(?P<value>\d{{1,2}}\.?\d{{3}}\.?\d{{3}}(?:-?[0-9Xx])?)(?!\d)"),
    "CRM": _compile(r"\bCRM(?:[/-]?\s*[A-Z]{2})?\s*(?:n[ºo°]\.?\s*)?(?P<value>\d{4,6})\b"),
    # a phone needs a DDD in parentheses, +55, a nine-digit mobile with hyphen, or a phone word
    "PHONE": _compile(
        r"(?<![\d-])(?P<value>(?:\+55\s?)?\(\d{2}\)\s?9?\d{4}-?\d{4})(?!\d)",
        r"(?<![\d-])(?P<value>\+55\s?\d{2}\s?9?\d{4}-?\d{4})(?!\d)",
        r"(?<![\d/.-])(?P<value>9\d{4}-\d{4})(?!\d)",
        rf"\b(?:tel(?:efone)?|fone|cel(?:ular)?|whats(?:app)?|contato)\b\.?{_CTX}"
        r"(?P<value>(?:\(\d{2}\)\s?)?9?\d{4}-?\d{4})(?!\d)",
        flags=re.IGNORECASE,
    ),
    "EMAIL": _compile(
        r"(?P<value>[A-Za-z0-9._%+-]+@(?P<domain>[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}))"
    ),
    "CEP": _compile(
        r"(?<![\d-])(?P<value>\d{5}-\d{3})(?!\d)",
        rf"\bCEP\b{_CTX}(?P<value>\d{{5}}-?\d{{3}})(?!\d)",
    ),
}


# --- fake pool ------------------------------------------------------------------------------


def _key(value: str) -> str:
    return re.sub(r"[^0-9X]", "", value.upper())


@dataclass(frozen=True)
class FakePool:
    cpf: frozenset[str]
    cnpj: frozenset[str]
    rg: frozenset[str]
    crm: frozenset[str]
    phone: frozenset[str]
    email_domains: frozenset[str]
    email_addresses: frozenset[str]


@cache
def load_pool() -> FakePool:
    raw = tomllib.loads(files("necrohub").joinpath("data/fake_identifiers.toml").read_text("utf-8"))

    def keys(name: str) -> frozenset[str]:
        return frozenset(_key(v) for v in raw[name])

    return FakePool(
        cpf=keys("cpf"),
        cnpj=keys("cnpj"),
        rg=keys("rg"),
        crm=keys("crm"),
        phone=keys("phone"),
        email_domains=frozenset(d.lower() for d in raw["email"]["domains"]),
        email_addresses=frozenset(a.lower() for a in raw["email"]["addresses"]),
    )


# --- leak check -----------------------------------------------------------------------------

_LOOSE_CPF = re.compile(r"(?<![\d.])\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
_LOOSE_CNPJ = re.compile(r"(?<![\d.])\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?!\d)")
_SKIP_NAMES = frozenset({"uv.lock", "dvc.lock"})


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    label: str
    value: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.label} ***{self.value[-2:]}"


def _phone_known(value: str, pool: FakePool) -> bool:
    key = _key(value)
    if key.startswith("55") and len(key) > 11:
        key = key[2:]
    return any(known.endswith(key) for known in pool.phone)


def scan_text(text: str, path: str, pool: FakePool | None = None) -> list[Finding]:
    """Findings for identifier-shaped values that are not in the fake pool."""
    pool = pool or load_pool()
    findings: list[Finding] = []
    for number, line in enumerate(text.split("\n"), start=1):
        found: list[tuple[str, str]] = []
        found += [
            ("CPF", m.group())
            for m in _LOOSE_CPF.finditer(line)
            if is_valid_cpf(m.group()) and _key(m.group()) not in pool.cpf
        ]
        found += [
            ("CNPJ", m.group())
            for m in _LOOSE_CNPJ.finditer(line)
            if is_valid_cnpj(m.group()) and _key(m.group()) not in pool.cnpj
        ]
        for label, known in (("RG", pool.rg), ("CRM", pool.crm)):
            found += [
                (label, m["value"])
                for p in IDENTIFIER_PATTERNS[label]
                for m in p.finditer(line)
                if _key(m["value"]) not in known
            ]
        found += [
            ("PHONE", m["value"])
            for p in IDENTIFIER_PATTERNS["PHONE"]
            for m in p.finditer(line)
            if not _phone_known(m["value"], pool)
        ]
        found += [
            ("EMAIL", m["value"])
            for m in IDENTIFIER_PATTERNS["EMAIL"][0].finditer(line)
            if m["domain"].lower() not in pool.email_domains
            and m["value"].lower() not in pool.email_addresses
        ]
        findings += [Finding(path, number, label, value) for label, value in found]
    return findings


def _tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return [root / name for name in out.split("\0") if name]


def collect_files(paths: Sequence[Path], root: Path) -> list[Path]:
    """Explicit files and directories, or else every tracked and not-ignored file under root."""
    if not paths:
        files_ = _tracked_files(root)
    else:
        files_ = [f for p in paths for f in ([p] if p.is_file() else sorted(p.rglob("*")))]
    keep = {
        f for f in files_ if f.is_file() and f.name not in _SKIP_NAMES and ".git" not in f.parts
    }
    return sorted(keep)


def check_files(files_: Sequence[Path], root: Path) -> list[Finding]:
    """Scan text files (binary and non-UTF-8 files are skipped)."""
    findings: list[Finding] = []
    base = root.resolve()
    for f in files_:
        data = f.read_bytes()
        if b"\0" in data[:8192]:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        resolved = f.resolve()
        shown = (
            resolved.relative_to(base).as_posix() if resolved.is_relative_to(base) else f.as_posix()
        )
        findings += scan_text(text, shown)
    return findings
