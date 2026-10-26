"""Static checks of the Dockerfile and compose.yaml (no docker needed)."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parents[2]
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}")


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "compose.yaml").read_text("utf-8"))  # type: ignore[no-any-return]


def test_dockerfile_is_pinned_non_root_with_a_healthcheck() -> None:
    lines = (ROOT / "Dockerfile").read_text("utf-8").splitlines()
    froms = [ln for ln in lines if ln.startswith("FROM ")]
    assert len(froms) == 2 and all(DIGEST.search(ln) for ln in froms)
    assert any(ln.startswith("COPY --from=ghcr.io/astral-sh/uv@sha256:") for ln in lines)
    assert any(ln.startswith("HEALTHCHECK") for ln in lines)
    users = [ln for ln in lines if ln.startswith("USER ")]
    assert users and users[-1] != "USER root"
    assert not any(ln.startswith("ENV") and "PASSWORD" in ln for ln in lines)


def test_compose_services_have_healthchecks_or_are_one_shot(compose: dict[str, Any]) -> None:
    services = compose["services"]
    assert set(services) == {"db", "loader", "api", "ui", "ollama", "ollama-pull"}
    for name, svc in services.items():
        if name in {"loader", "ollama-pull"}:
            assert svc["restart"] == "no"
        else:
            assert svc["healthcheck"]["test"], name
    for name in ("ollama", "ollama-pull"):
        assert services[name]["profiles"] == ["ollama"]


def test_compose_external_images_are_pinned_by_digest(compose: dict[str, Any]) -> None:
    for name in ("db", "ollama", "ollama-pull"):
        assert DIGEST.search(compose["services"][name]["image"]), name


def test_compose_ports_are_loopback_only_and_secrets_come_from_env(compose: dict[str, Any]) -> None:
    for svc in compose["services"].values():
        assert all(p.startswith("127.0.0.1:") for p in svc.get("ports", []))
    text = (ROOT / "compose.yaml").read_text("utf-8")
    for match in re.finditer(r"^\s*\w*PASSWORD:\s*(\S+)", text, re.MULTILINE):
        assert match.group(1).startswith("${"), match.group(0)
    assert "${POSTGRES_PASSWORD:?" in text
