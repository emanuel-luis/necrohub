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
