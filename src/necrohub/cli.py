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
