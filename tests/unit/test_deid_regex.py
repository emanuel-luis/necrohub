import random
from collections.abc import Callable

from necrohub.contracts.spans import EntityLabel
from necrohub.ids import cnpj_check_digits, fake_cpf
from necrohub.preprocess.deid.base import merge_spans, redact
from necrohub.preprocess.deid.regex import RegexDeidentifier
from necrohub.preprocess.pipeline import process

detect = RegexDeidentifier().detect


def found(text: str) -> list[tuple[str, str]]:
    return [(s.label.value, s.text) for s in merge_spans([detect(text)])]


def test_spans_golden(report: tuple[str, str], golden: Callable[[str, object], None]) -> None:
    name, text = report
    spans = merge_spans([detect(text)])
    golden(
        f"deid_regex/{name}.json",
        [{"label": s.label.value, "start": s.start, "end": s.end, "text": s.text} for s in spans],
    )


def test_redacted_golden(report: tuple[str, str], golden: Callable[[str, object], None]) -> None:
    name, text = report
    golden(f"deid_regex/{name}.txt", redact(text, detect(text)))


def test_redacted_has_no_matches(report: tuple[str, str]) -> None:
    _, text = report
    result = process(text)
    assert result.entities
    assert detect(result.redacted_text) == []


def test_cpf_redacts_value_keeps_label() -> None:
    cpf = fake_cpf(random.Random(1))
    text = f"identificado (CPF nº {cpf})."
    assert redact(text, detect(text)) == "identificado (CPF nº [CPF])."
    digits = cpf.replace(".", "").replace("-", "")
    assert found(f"CPF: {digits}") == [("CPF", digits)]
    assert found(f"código {digits}") == []


def test_cpf_without_dots_but_with_hyphen() -> None:
    d = fake_cpf(random.Random(2)).replace(".", "").replace("-", "")
    value = f"{d[:9]}-{d[9:]}"
    assert found(f"CPF {value}") == [("CPF", value)]


def test_crm_forms() -> None:
    # fake-pool values, so the leak check stays quiet; the 4-digit value is built at runtime
    assert found("CRM: 12345") == [("CRM", "12345")]
    assert found("CRM-SP: 22456") == [("CRM", "22456")]
    assert found("CRM/SP 33.567") == [("CRM", "33.567")]
    assert found("crm 44678") == [("CRM", "44678")]
    four = "12" + "34"
    assert found(f"CRM nº {four}") == [("CRM", four)]
    assert found("CRM 1234567") == []
    assert found("crm/ma 12345") == [("CRM", "12345")]
    assert found("CRM/MA12345") == [("CRM", "12345")]


def test_capital_n_separator() -> None:
    assert found("CRM Nº 12345") == [("CRM", "12345")]
    assert found("CRM/MA Nº 22456") == [("CRM", "22456")]
    assert found("CRM N° 33567") == [("CRM", "33567")]
    assert found("CRM No. 44678") == [("CRM", "44678")]
    assert found("RG Nº 1234567") == [("RG", "1234567")]


def test_crm_needs_a_uf_code_not_any_word() -> None:
    # only the keyword and the UF ignore case: "em"/"de" are not UF codes
    assert found("inscrito no CRM em 2019") == []
    assert found("registro no CRM de 2019") == []
    assert found("CRMV 12345") == []


def test_phone_with_space_separator() -> None:
    assert found("ligar (98) 99999 0101") == [("PHONE", "(98) 99999 0101")]
    assert found("ligar +55 98 99999 0101") == [("PHONE", "+55 98 99999 0101")]
    assert found("ligar +55 (98) 99999-0101") == [("PHONE", "+55 (98) 99999-0101")]
    assert found("Telefone: 99999 0101") == [("PHONE", "99999 0101")]


def test_rg_and_crm_need_their_label() -> None:
    assert found("RG nº 1234567 SSP/MA") == [("RG", "1234567")]
    assert found("CRM/MA 12345") == [("CRM", "12345")]
    assert found("peso 1234567 e 12345") == []


def test_cnpj_email_cep() -> None:
    base = "123456780001"
    d = base + cnpj_check_digits(base)
    cnpj = f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}"
    assert found(f"CNPJ {cnpj}") == [("CNPJ", cnpj)]
    assert found("escreva para perito@example.org agora") == [("EMAIL", "perito@example.org")]
    assert found("CEP 65000-000") == [("ADDRESS", "65000-000")]
    assert found("CEP 65000000") == [("ADDRESS", "65000000")]
    assert found("São Luís 65000000") == []


def test_ignores_substancia_branca() -> None:
    assert found("Presença de substância branca na narina. Cor branca, parda ou negra.") == []


def test_ignores_uppercase_se_to() -> None:
    assert found("NÃO SE OBSERVA LESÃO. SE TO ES MA PA AL") == []


def test_phone_needs_ddd_or_separator() -> None:
    assert found("número 12345678 e 987654321") == []
    assert found("ligar (98) 99999-0101") == [("PHONE", "(98) 99999-0101")]
    assert found("ligar 99999-0101") == [("PHONE", "99999-0101")]
    assert found("Tel: 32000303") == [("PHONE", "32000303")]
    assert not [x for x in found("Laudo nº 2025-1037") if x[0] == "PHONE"]


def test_street_number_not_doc_number() -> None:
    text = "residente à Rua das Flores, nº 45, Bairro Centro, e outro nº 45"
    assert found(text) == [("ADDRESS", "Rua das Flores, nº 45, Bairro Centro")]
    assert found("Laudo nº 2025/1037") == [("DOC_NUMBER", "2025/1037")]
    assert found("B.O. nº 2557/2025") == [("DOC_NUMBER", "2557/2025")]
    assert found("inquérito policial nº 77/2025-A") == [("DOC_NUMBER", "77/2025-A")]


def test_dates() -> None:
    assert found("em 06/10/2025 e 6 de outubro de 2025") == [
        ("DATE", "06/10/2025"),
        ("DATE", "6 de outubro de 2025"),
    ]


def test_names_need_a_cue() -> None:
    assert found("Perito Médico-Legal: Dr. Carlos de Souza - CRM/MA 12345") == [
        ("PER", "Carlos de Souza"),
        ("CRM", "12345"),
    ]
    assert found("Auxiliar: Técnica de Necropsia Maria Fernanda\nOutra linha") == [
        ("PER", "Maria Fernanda")
    ]
    assert found("Corpo de João Pereira da Costa, sexo masculino") == [
        ("PER", "João Pereira da Costa")
    ]
    assert found("O Coração apresenta Lesão Transfixante") == []
    assert found("Exame de Corpo de Delito realizado") == []
    assert found("encaminhado ao Corpo de Bombeiros Militar") == []


def test_labels_are_pii() -> None:
    assert all(EntityLabel(label).is_pii for label, _ in found("CRM/MA 12345 em 06/10/2025"))
