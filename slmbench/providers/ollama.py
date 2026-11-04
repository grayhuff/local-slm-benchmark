"""Small helpers for Ollama's native REST API.

Inference goes through the OpenAI-compatible ``/v1`` endpoint; these native
calls only answer questions that endpoint can't: what's installed, what's
loaded, and how much (V)RAM it is using.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class OllamaModel:
    name: str
    size_gb: float
    parameter_size: str = ""
    quantization: str = ""
    family: str = ""


@dataclass
class OllamaStatus:
    running: bool
    base_url: str
    version: str | None = None
    models: list[OllamaModel] = field(default_factory=list)
    error: str | None = None


def get_status(base_url: str, timeout: float = 1.5) -> OllamaStatus:
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            tags = client.get("/api/tags")
            tags.raise_for_status()
            try:
                version = client.get("/api/version").json().get("version")
            except (httpx.HTTPError, ValueError):
                version = None
    except httpx.ConnectError:
        return OllamaStatus(False, base_url, error=f"Ollama isn't running at {base_url}")
    except httpx.HTTPError as exc:
        return OllamaStatus(False, base_url, error=f"Ollama at {base_url} returned an error: {exc}")

    models = []
    for m in tags.json().get("models", []):
        details = m.get("details") or {}
        models.append(
            OllamaModel(
                name=m.get("name") or m.get("model", "?"),
                size_gb=round((m.get("size") or 0) / 1e9, 2),
                parameter_size=details.get("parameter_size", ""),
                quantization=details.get("quantization_level", ""),
                family=details.get("family", ""),
            )
        )
    models.sort(key=lambda m: m.name)
    return OllamaStatus(True, base_url, version=version, models=models)


def loaded_models(base_url: str, timeout: float = 1.5) -> list[dict[str, Any]]:
    """Models currently resident in memory (``/api/ps``), with ``size`` and ``size_vram`` in bytes."""
    try:
        resp = httpx.get(f"{base_url}/api/ps", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return []
