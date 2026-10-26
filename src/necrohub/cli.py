from pathlib import Path
from typing import Annotated

import typer

from necrohub import __version__

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Synthetic necropsy report pipeline."
)
index_app = typer.Typer(
    no_args_is_help=True, help="Build the serving index or load it into pgvector."
)
app.add_typer(index_app, name="index")


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


@app.command("export-sft")
def export_sft(
    params: Annotated[Path, typer.Option(help="Pipeline parameters file.")] = Path("params.yaml"),
    data: Annotated[Path, typer.Option(help="Datasets layer directory.")] = Path("data/datasets"),
    out: Annotated[Path, typer.Option(help="SFT directory to (re)write.")] = Path("data/sft"),
    runs_dir: Annotated[Path, typer.Option(help="Where run manifests go.")] = Path("runs"),
    metrics_dir: Annotated[Path, typer.Option(help="Where metrics/export_sft.json goes.")] = Path(
        "metrics"
    ),
) -> None:
    """Export chat-format SFT examples and the extraction JSON schema."""
    from necrohub.params import load_params
    from necrohub.runs import StageFailed
    from necrohub.sft.export import run_export_sft

    try:
        run = run_export_sft(
            load_params(params), data, out, runs_dir=runs_dir, metrics_dir=metrics_dir
        )
    except (StageFailed, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"export-sft: {run.counts.written} examples in {out.as_posix()} "
        f"({run.counts.rejected} rejected, run {run.run_id})"
    )


@app.command("eval-extraction")
def eval_extraction(
    predictions: Annotated[
        Path, typer.Option(help='JSONL of {"id", "text"}: the raw model output per record.')
    ],
    data: Annotated[Path, typer.Option(help="Datasets layer directory.")] = Path("data/datasets"),
    split: Annotated[str, typer.Option(help="train, validation or test.")] = "test",
    out: Annotated[Path, typer.Option(help="Results file to write.")] = Path("results.json"),
) -> None:
    """Score raw model output against the extraction gold of a split."""
    from necrohub.evaluate.extraction import score_predictions_file

    if split not in ("train", "validation", "test"):
        typer.echo(f"unknown split: {split}", err=True)
        raise typer.Exit(2)
    try:
        scores = score_predictions_file(data, split, predictions, out)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"eval-extraction: {scores['n']} records, valid {scores['valid_json_rate']}, "
        f"all fields exact {scores['all_fields_exact']} ({out.as_posix()})"
    )


@app.command("check-repro")
def check_repro(
    baseline_lock: Annotated[Path, typer.Option(help="A copy of the committed dvc.lock.")],
    baseline_metrics: Annotated[Path, typer.Option(help="A copy of the committed metrics/.")],
    lock: Annotated[Path, typer.Option(help="The dvc.lock a fresh run wrote.")] = Path("dvc.lock"),
    metrics: Annotated[Path, typer.Option(help="The metrics/ a fresh run wrote.")] = Path(
        "metrics"
    ),
    abs_tol: Annotated[float, typer.Option(help="Float warning threshold, model stages.")] = 0.05,
    rel_tol: Annotated[float, typer.Option(help="Relative float warning threshold.")] = 0.05,
    count_tol: Annotated[
        int, typer.Option(help="Warning threshold on tp, fp, fn and missed PII counts.")
    ] = 2,
    strict_models: Annotated[
        bool,
        typer.Option(help="Fail on model-determined drift too (for a rerun on the same machine)."),
    ] = False,
) -> None:
    """Fail unless data stages reproduce bit for bit and model stages keep their inputs, structure
    and data-determined fields. Model-determined drift is a warning (an error with
    --strict-models)."""
    import os

    from necrohub.repro import check_repro as compare

    result = compare(
        baseline_lock,
        baseline_metrics,
        lock,
        metrics,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        count_tol=count_tol,
        strict_models=strict_models,
    )
    for error in result.errors:
        typer.echo(error)
    actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for warning in result.warnings:
        typer.echo(f"warning: {warning}")
        if actions:
            typer.echo(f"::warning::{warning}")
    if result.warnings:
        typer.echo(f"{len(result.warnings)} model-determined difference(s) beyond tolerance")
    if result.errors:
        typer.echo(f"{len(result.errors)} difference(s) from the baseline", err=True)
        raise typer.Exit(1)
    typer.echo("reproduced: data stages identical, model stages keep inputs and structure")


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


@index_app.command("build")
def index_build(
    params: Annotated[Path, typer.Option(help="Pipeline parameters file.")] = Path("params.yaml"),
    in_: Annotated[Path, typer.Option("--in", help="Processed layer directory.")] = Path(
        "data/processed"
    ),
    gold: Annotated[Path, typer.Option(help="Raw gold records (the leak metric only).")] = Path(
        "data/raw/gold"
    ),
    out: Annotated[Path, typer.Option(help="Index directory to (re)write.")] = Path(
        "data/serving/index"
    ),
    runs_dir: Annotated[Path, typer.Option(help="Where run manifests go.")] = Path("runs"),
    metrics_dir: Annotated[Path, typer.Option(help="Where metrics/index.json goes.")] = Path(
        "metrics"
    ),
) -> None:
    """Chunk the redacted text of the processed layer, embed it and write the index files."""
    from necrohub.index.build import run_index_build
    from necrohub.params import load_params
    from necrohub.runs import StageFailed

    try:
        run = run_index_build(
            load_params(params), in_, gold, out, runs_dir=runs_dir, metrics_dir=metrics_dir
        )
    except (StageFailed, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"index build: {run.counts.written} chunks in {out.as_posix()} (run {run.run_id})")


def _error_line(exc: BaseException) -> str:
    from necrohub.runs import error_line

    return error_line(exc)


def _db_errors() -> tuple[type[Exception], ...]:
    try:
        import psycopg
    except ImportError:
        return ()
    return (psycopg.Error,)


@index_app.command("load")
def index_load(
    in_: Annotated[Path, typer.Option("--in", help="Index directory.")] = Path(
        "data/serving/index"
    ),
    runs_dir: Annotated[Path, typer.Option(help="Where run manifests go.")] = Path("runs"),
) -> None:
    """Upsert the index into pgvector (DATABASE_URL), keyed by its index_version."""
    import psycopg

    from necrohub.index.load import run_index_load
    from necrohub.index.pgvector import PgVectorStore
    from necrohub.runs import StageFailed
    from necrohub.settings import Settings

    dsn = Settings().database_url
    if dsn is None:
        typer.echo("DATABASE_URL is not set", err=True)
        raise typer.Exit(1)
    try:
        run = run_index_load(in_, PgVectorStore(dsn), runs_dir=runs_dir)
    except (StageFailed, ValueError, OSError, psycopg.Error) as exc:
        typer.echo(_error_line(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"index load: {run.counts.written} chunks stored (run {run.run_id})")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="The question, in Portuguese.")],
    store: Annotated[
        str | None, typer.Option(help="memory or pgvector. Default: NECROHUB_STORE or memory.")
    ] = None,
    index: Annotated[
        Path | None, typer.Option(help="Index directory. Default: NECROHUB_INDEX_DIR.")
    ] = None,
    k: Annotated[int, typer.Option(min=1, max=20, help="Chunks to retrieve.")] = 4,
    params: Annotated[
        Path | None,
        typer.Option(
            help="Pipeline parameters file (embed). Default: NECROHUB_PARAMS or params.yaml."
        ),
    ] = None,
) -> None:
    """Answer a question from the serving index and print the answer with its citations."""
    from necrohub.embed.factory import build_embedder
    from necrohub.llm.base import LlmError
    from necrohub.params import load_params
    from necrohub.serve.rag import answer, case_id_in, llm_from_settings
    from necrohub.serve.runtime import ConfigError, open_store
    from necrohub.settings import Settings

    if store not in (None, "memory", "pgvector"):
        typer.echo(f"unknown store: {store} (memory or pgvector)", err=True)
        raise typer.Exit(2)
    overrides: dict[str, object] = {}
    if params:
        overrides["params_path"] = params
    if store:
        overrides["store"] = store
    if index:
        overrides["index_dir"] = index
    settings = Settings(**overrides)  # type: ignore[arg-type]
    caught: tuple[type[Exception], ...] = (ConfigError, LlmError, ValueError, OSError)
    caught += _db_errors()
    try:
        embedder = build_embedder(load_params(settings.params_path).embed)
        vector_store, version, _ = open_store(settings, embedder.id)
        result = answer(
            question,
            embedder=embedder,
            store=vector_store,
            index_version=version,
            llm=llm_from_settings(settings),
            k=k,
        )
        scores = {
            h.chunk.chunk_id: h.score
            for h in vector_store.search(
                version, embedder.embed([question])[0], k, doc_id=case_id_in(question)
            )
        }
    except caught as exc:
        typer.echo(_error_line(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Pergunta: {result.question}")
    typer.echo(f"Resposta ({result.backend}): {result.answer}")
    typer.echo(f"Fontes (index {result.index_version}):")
    for n, c in enumerate(result.citations, 1):
        score = scores.get(c.chunk_id)
        shown = f"{score:.4f}" if score is not None else "n/a"
        typer.echo(
            f"  [{n}] doc_id={c.doc_id} section={c.section} ordinal={c.ordinal} score={shown}"
        )


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
) -> None:
    """Serve GET /healthz and POST /ask (settings from NECROHUB_* and DATABASE_URL)."""
    import uvicorn

    from necrohub.llm.base import LlmError
    from necrohub.serve.api import create_app
    from necrohub.serve.runtime import ConfigError
    from necrohub.settings import Settings

    caught: tuple[type[Exception], ...] = (ConfigError, LlmError, ValueError, OSError)
    caught += _db_errors()
    try:
        api = create_app(Settings())
    except caught as exc:
        typer.echo(_error_line(exc), err=True)
        raise typer.Exit(1) from exc
    uvicorn.run(api, host=host, port=port)


@app.command()
def ui(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8501,
) -> None:
    """Run the Streamlit UI (`ui` extra); it calls the API at NECROHUB_API_URL."""
    import importlib.util
    import subprocess
    import sys

    spec = importlib.util.find_spec("necrohub.serve.ui")
    if spec is None or spec.origin is None:
        raise typer.Exit(1)
    argv = [sys.executable, "-m", "streamlit", "run", spec.origin]
    argv += ["--server.address", host, "--server.port", str(port), "--server.headless", "true"]
    raise typer.Exit(subprocess.call(argv))
