import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.ids import cpf_check_digits

runner = CliRunner()


def valid_cpf() -> str:
    base = "390533447"  # arbitrary base: the check digits are computed, never written out
    d = base + cpf_check_digits(base)
    return f"{d[:3]}.{d[3:6]}.{d[6:9]}-{d[9:]}"


def check(path: Path) -> tuple[int, str]:
    result = runner.invoke(app, ["check-identifiers", str(path)])
    return result.exit_code, result.output


def test_valid_cpf_is_flagged_and_masked(tmp_path: Path) -> None:
    cpf = valid_cpf()
    f = tmp_path / "a.txt"
    f.write_text(f"ok\nCPF nº {cpf}\n", encoding="utf-8")
    code, out = check(f)
    assert code == 1
    assert f"a.txt:2: CPF ***{cpf[-2:]}" in out.replace("\\", "/")
    assert cpf not in out and cpf[:3] not in out


def test_unformatted_valid_cpf_is_flagged(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text(valid_cpf().replace(".", "").replace("-", ""), encoding="utf-8")
    assert check(f)[0] == 1


def test_allowlisted_values_pass(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text(
        "RG nº 125623240 SSP/MA, CRM/MA 12345, CPF 123.456.789-00, tel (98) 99999-0101\n"
        "perito@example.org, emanuel.filho08@gmail.com, 1234+bot@users.noreply.github.com\n",
        encoding="utf-8",
    )
    assert check(f)[0] == 0


def test_unknown_rg_crm_phone_email_are_flagged(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text(
        f"RG nº {'9' * 7}\nCRM/MA {'8' * 5}\nfone ({'9' * 2}) 9{'7' * 4}-{'1' * 4}\n"
        + "ana@"
        + "acme.test\n",
        encoding="utf-8",
    )
    code, out = check(f)
    assert code == 1
    for label in ("RG", "CRM", "PHONE", "EMAIL"):
        assert f": {label} ***" in out


def test_lock_files_and_binaries_are_skipped(tmp_path: Path) -> None:
    for name in ("uv.lock", "dvc.lock"):
        (tmp_path / name).write_text(valid_cpf(), encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\0" + valid_cpf().encode())
    assert check(tmp_path)[0] == 0


def test_cp1252_file_is_scanned(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_bytes(f"necropsia ação\nCPF nº {valid_cpf()}\n".encode("cp1252"))
    assert check(f)[0] == 1


@pytest.mark.parametrize("codec", ["utf-16", "utf-32", "utf-8-sig"])
def test_bom_files_are_scanned(tmp_path: Path, codec: str) -> None:
    f = tmp_path / "a.txt"
    f.write_bytes(f"ação\nCPF nº {valid_cpf()}\n".encode(codec))
    assert check(f)[0] == 1


@pytest.mark.parametrize("codec", ["utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"])
def test_bomless_wide_files_are_scanned(tmp_path: Path, codec: str) -> None:
    f = tmp_path / "a.txt"
    f.write_bytes(f"necropsia ação\nCPF nº {valid_cpf()}\n".encode(codec))
    assert check(f)[0] == 1


def test_crm_in_prose_is_not_a_finding(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("inscrito no CRM em 2019\nregistro no CRM de 2019\n", encoding="utf-8")
    assert check(f)[0] == 0


def test_binary_file_is_reported_as_skipped(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\0" + valid_cpf().encode())
    code, out = check(tmp_path)
    assert code == 0
    assert "1 binary skipped (" in out and "blob.bin" in out
    assert valid_cpf() not in out


def test_crm_phone_forms_are_flagged(tmp_path: Path) -> None:
    crm, dotted = "8" * 5, f"{'7' * 3}.{'6' * 3}"
    phone = f"9{'7' * 4} {'1' * 4}"
    forms = [
        f"CRM: {crm}",
        f"CRM-SP: {crm}",
        f"CRM/SP {dotted}",
        f"crm {crm}",
        f"({'9' * 2}) {phone}",
        f"+55 {'9' * 2} {phone}",
    ]
    for i, form in enumerate(forms):
        f = tmp_path / f"f{i}.txt"
        f.write_text(form, encoding="utf-8")
        assert check(f)[0] == 1, form


def test_default_file_set_is_git_tracked_and_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text(valid_cpf(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["check-identifiers"]).exit_code == 0
    (tmp_path / "new.txt").write_text(valid_cpf(), encoding="utf-8")
    result = runner.invoke(app, ["check-identifiers"])
    assert result.exit_code == 1 and "new.txt:1: CPF" in result.output
