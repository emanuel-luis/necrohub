from pathlib import Path
from typing import Annotated

import typer

from necrohub import __version__

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Synthetic necropsy report pipeline."
)


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    pass


@app.command("check-identifiers")
def check_identifiers(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Files or directories. Default: every git-tracked or new file."),
    ] = None,
) -> None:
    """Fail when a file holds an identifier that is not in the fake pool."""
    from necrohub.ids import check_files, collect_files

    root = Path.cwd()
    files = collect_files(paths or [], root)
    findings, skipped = check_files(files, root)
    for finding in findings:
        typer.echo(finding)
    note = f", {len(skipped)} binary skipped ({', '.join(skipped)})" if skipped else ""
    if findings:
        typer.echo(f"{len(findings)} finding(s) in {len(files)} files{note}", err=True)
        raise typer.Exit(1)
    typer.echo(f"checked {len(files)} files: no leaked identifiers{note}")


@app.command()
def generate(
    params: Annotated[Path, typer.Option(help="Pipeline parameters file.")] = Path("params.yaml"),
    out: Annotated[Path, typer.Option(help="Raw layer directory to (re)write.")] = Path("data/raw"),
    runs_dir: Annotated[Path, typer.Option(help="Where run manifests go.")] = Path("runs"),
    metrics_dir: Annotated[Path, typer.Option(help="Where metrics/generate.json goes.")] = Path(
        "metrics"
    ),
) -> None:
    """Generate synthetic reports with gold labels into the raw layer."""
    from necrohub.generate.stage import run_generate
    from necrohub.params import load_params
    from necrohub.runs import StageFailed

    try:
        run = run_generate(load_params(params), out, runs_dir=runs_dir, metrics_dir=metrics_dir)
    except StageFailed as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"generate: {run.counts.written} reports in {out.as_posix()} (run {run.run_id})")


@app.command()
def ingest(
    params: Annotated[Path, typer.Option(help="Pipeline parameters file.")] = Path("params.yaml"),
    in_: Annotated[Path, typer.Option("--in", help="Raw layer directory.")] = Path("data/raw"),
    out: Annotated[Path, typer.Option(help="Interim layer directory to (re)write.")] = Path(
        "data/interim"
    ),
    runs_dir: Annotated[Path, typer.Option(help="Where run manifests go.")] = Path("runs"),
    metrics_dir: Annotated[Path, typer.Option(help="Where metrics/ingest.json goes.")] = Path(
        "metrics"
    ),
) -> None:
    """Read every raw report (PDF text when the case has a PDF) into the interim layer."""
    from necrohub.ingest.stage import run_ingest
    from necrohub.params import load_params
    from necrohub.runs import StageFailed

    try:
        run = run_ingest(
            load_params(params).ingest, in_, out, runs_dir=runs_dir, metrics_dir=metrics_dir
        )
    except StageFailed as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"ingest: {run.counts.written} documents in {out.as_posix()} (run {run.run_id})")


@app.command()
def preprocess(
    params: Annotated[Path, typer.Option(help="Pipeline parameters file.")] = Path("params.yaml"),
    in_: Annotated[Path, typer.Option("--in", help="Interim layer directory.")] = Path(
        "data/interim"
    ),
    out: Annotated[Path, typer.Option(help="Processed layer directory to (re)write.")] = Path(
        "data/processed"
    ),
    raw: Annotated[
        Path | None,
        typer.Option(
            help="Raw layer with gold/ and reports/ for the alignment metric. "
            "Default: raw next to --in."
        ),
    ] = None,
    runs_dir: Annotated[Path, typer.Option(help="Where run manifests go.")] = Path("runs"),
    metrics_dir: Annotated[Path, typer.Option(help="Where metrics/preprocess.json goes.")] = Path(
        "metrics"
    ),
) -> None:
    """Normalize, split into sections and de-identify every ingested document."""
    from necrohub.params import load_params
    from necrohub.preprocess.stage import run_preprocess
    from necrohub.runs import StageFailed

    try:
        run = run_preprocess(
            load_params(params).preprocess,
            in_,
            out,
            raw=raw,
            runs_dir=runs_dir,
            metrics_dir=metrics_dir,
        )
    except StageFailed as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"preprocess: {run.counts.written} documents in {out.as_posix()} (run {run.run_id})")


@app.command("build-dataset")
def build_dataset(
    params: Annotated[Path, typer.Option(help="Pipeline parameters file.")] = Path("params.yaml"),
    processed: Annotated[Path, typer.Option(help="Processed layer directory.")] = Path(
        "data/processed"
    ),
    raw: Annotated[Path, typer.Option(help="Raw layer with gold/ and reports/.")] = Path(
        "data/raw"
    ),
    out: Annotated[Path, typer.Option(help="Datasets layer directory to (re)write.")] = Path(
        "data/datasets"
    ),
    runs_dir: Annotated[Path, typer.Option(help="Where run manifests go.")] = Path("runs"),
    metrics_dir: Annotated[
        Path, typer.Option(help="Where metrics/build_dataset.json goes.")
    ] = Path("metrics"),
) -> None:
    """Build the per-task train/validation/test datasets and the dataset card."""
    from necrohub.datasets.stage import run_build_dataset
    from necrohub.params import load_params
    from necrohub.runs import StageFailed

    try:
        run = run_build_dataset(
            load_params(params), processed, raw, out, runs_dir=runs_dir, metrics_dir=metrics_dir
        )
    except StageFailed as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"build-dataset: {run.counts.written} records in {out.as_posix()} (run {run.run_id})"
    )


@app.command("check-repro")
def check_repro(
    baseline_lock: Annotated[Path, typer.Option(help="A copy of the committed dvc.lock.")],
    baseline_metrics: Annotated[Path, typer.Option(help="A copy of the committed metrics/.")],
    lock: Annotated[Path, typer.Option(help="The dvc.lock a fresh run wrote.")] = Path("dvc.lock"),
    metrics: Annotated[Path, typer.Option(help="The metrics/ a fresh run wrote.")] = Path(
        "metrics"
    ),
    abs_tol: Annotated[float, typer.Option(help="Float tolerance, model stages.")] = 0.05,
    rel_tol: Annotated[float, typer.Option(help="Relative float tolerance, model stages.")] = 0.05,
    count_tol: Annotated[
        int, typer.Option(help="Tolerance on tp, fp, fn and missed PII counts, model stages.")
    ] = 2,
) -> None:
    """Fail unless data stages reproduce bit for bit and model stages within tolerance."""
    from necrohub.repro import check_repro as compare

    diffs = compare(
        baseline_lock, baseline_metrics, lock, metrics, rel_tol=rel_tol, abs_tol=abs_tol
    )
    for diff in diffs:
        typer.echo(diff)
    if diffs:
        typer.echo(f"{len(diffs)} difference(s) from the baseline", err=True)
        raise typer.Exit(1)
    typer.echo("reproduced: data stages identical, model stages within tolerance")


@app.command()
def train(
    task: Annotated[str, typer.Option(help="ner or sections.")],
    params: Annotated[Path, typer.Option(help="Pipeline parameters file.")] = Path("params.yaml"),
    data: Annotated[Path, typer.Option(help="Datasets layer directory.")] = Path("data/datasets"),
    out: Annotated[
        Path | None, typer.Option(help="Model directory. Default models/encoder-TASK.")
    ] = None,
    runs_dir: Annotated[Path, typer.Option(help="Where run manifests go.")] = Path("runs"),
    metrics_dir: Annotated[Path, typer.Option(help="Where metrics/train_TASK.json goes.")] = Path(
        "metrics"
    ),
) -> None:
    """Fine-tune the encoder (train.base_model) for one task on the train split."""
    from necrohub.params import load_params
    from necrohub.runs import StageFailed
    from necrohub.train.stage import run_train

    if task not in ("ner", "sections"):
        typer.echo(f"unknown task: {task} (ner or sections)", err=True)
        raise typer.Exit(2)
    target = out or Path("models") / f"encoder-{task}"
    try:
        run = run_train(
            task,  # type: ignore[arg-type]
            load_params(params),
            data,
            target,
            metrics_dir=metrics_dir,
            runs_dir=runs_dir,
        )
    except (StageFailed, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"train: {task} model in {target.as_posix()} (run {run.run_id})")


@app.command("gpu-check")
def gpu_check() -> None:
    """Report the torch CUDA build and fail unless it has kernels for the installed GPU."""
    from necrohub.train.gpu import gpu_problem, gpu_report

    report = gpu_report()
    for key, value in report.items():
        typer.echo(f"{key}: {value}")
    problem = gpu_problem(report)
    if problem:
        typer.echo(problem, err=True)
        raise typer.Exit(1)
    typer.echo("gpu-check: ok")


@app.command()
def evaluate(
    params: Annotated[Path, typer.Option(help="Pipeline parameters file.")] = Path("params.yaml"),
    models: Annotated[
        Path, typer.Option(help="Directory with encoder-ner and encoder-sections.")
    ] = Path("models"),
    data: Annotated[Path, typer.Option(help="Datasets layer directory.")] = Path("data/datasets"),
    raw: Annotated[Path, typer.Option(help="Raw layer (gold templates).")] = Path("data/raw"),
    out: Annotated[Path, typer.Option(help="Metrics file to write.")] = Path("metrics/eval.json"),
    runs_dir: Annotated[Path, typer.Option(help="Where run manifests go.")] = Path("runs"),
) -> None:
    """Score the fine-tuned encoders and the regex baseline on the test split."""
    from necrohub.evaluate.stage import run_evaluate
    from necrohub.params import load_params
    from necrohub.runs import StageFailed

    try:
        run = run_evaluate(load_params(params), models, data, raw, out, runs_dir=runs_dir)
    except (StageFailed, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"evaluate: {out.as_posix()} (run {run.run_id})")
