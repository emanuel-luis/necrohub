# App image for the api, ui and loader services (compose.yaml). No torch, no ml extra.
# python:3.12-slim-bookworm, pinned by the digest of its multi-arch index (Docker Hub registry API)
FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS builder
# uv 0.13.5 from the official image, pinned by digest
COPY --from=ghcr.io/astral-sh/uv@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
ARG EXTRAS="--extra serve --extra db --extra ui --extra ollama"
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
# hadolint ignore=SC2086 - EXTRAS is a list of flags, word splitting is intended
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable ${EXTRAS}

FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70
RUN groupadd --gid 10001 necrohub && useradd --uid 10001 --gid 10001 --create-home necrohub
WORKDIR /app
# stage_run writes runs/<stage>/<id>.json under the working directory
RUN mkdir /app/runs && chown 10001:10001 /app/runs
COPY --from=builder /app/.venv /app/.venv
COPY params.yaml ./
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 NECROHUB_PARAMS=/app/params.yaml
USER 10001:10001
EXPOSE 8000
# the ui and loader services override this in compose.yaml
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
CMD ["necrohub", "serve", "--host", "0.0.0.0", "--port", "8000"]
