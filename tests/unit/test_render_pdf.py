from pathlib import Path

from necrohub.generate import stage
from necrohub.generate.render_pdf import render_pdf, transliterate, wants_pdf
from necrohub.params import GenerateParams, Params
from necrohub.runs import sha256_tree

TEXT = "Laudo de Exame de Necropsia\n\nConclusão: asfixia.\n" * 40


def test_rendering_twice_gives_identical_bytes() -> None:
    a, n = render_pdf(TEXT)
    assert a.startswith(b"%PDF-") and n == 0
    assert render_pdf(TEXT)[0] == a


def test_non_latin1_input_is_transliterated_and_counted() -> None:
    assert transliterate("a — “b”…") == ('a - "b"...', 4)
    data, n = render_pdf("a — b 一\n")
    assert data.startswith(b"%PDF-") and n == 2  # one dash transliterated, one char replaced


def test_pdf_selection_is_deterministic_and_follows_the_fraction() -> None:
    ids = [f"c7-{i:05d}" for i in range(400)]
    picks = [wants_pdf(7, c, 0.25) for c in ids]
    assert picks == [wants_pdf(7, c, 0.25) for c in ids]
    assert abs(sum(picks) / 400 - 0.25) < 0.06
    assert not any(wants_pdf(7, c, 0.0) for c in ids) and all(wants_pdf(7, c, 1.0) for c in ids)


def test_generate_with_pdfs_is_byte_identical(tmp_path: Path) -> None:
    params = Params(
        seed=7,
        generate=GenerateParams(
            n_cases=8,
            realizer="template",
            holdout_templates=["t3"],
            max_reject_fraction=0.2,
            pdf_fraction=0.5,
        ),
    )
    trees = []
    for name in ("a", "b"):
        out = tmp_path / name
        stage.run_generate(params, out, runs_dir=tmp_path / "runs", metrics_dir=tmp_path / "m")
        trees.append(sha256_tree(out))
    pdfs = sorted(p.name for p in (tmp_path / "a" / "reports").glob("*.pdf"))
    assert pdfs and trees[0] == trees[1]
