"""Paths, environment loading and provider presets.

Every provider except Anthropic is reached through the OpenAI-compatible
Chat Completions API, so adding a new one is usually a single preset entry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICING_PATH = PROJECT_ROOT / "config" / "pricing.json"
BUILTIN_SUITE_PATH = PROJECT_ROOT / "data" / "builtin_suite.jsonl"

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def load_env() -> None:
    """Load ``.env`` from the project root without overriding real env vars."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def runs_dir() -> Path:
    return Path(os.getenv("SLMBENCH_RUNS_DIR", PROJECT_ROOT / "runs"))


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def is_local_url(url: str | None) -> bool:
    if not url:
        return False
    host = urlparse(url if "://" in url else f"http://{url}").hostname or ""
    return host in LOCAL_HOSTS or host.endswith(".localhost")


@dataclass(frozen=True)
class ProviderPreset:
    id: str
    name: str
    kind: Literal["openai_compat", "anthropic", "simulated"]
    base_url: str | None
    api_key_envs: tuple[str, ...]  # first one set wins
    local: bool
    # OpenAI's newer models only accept max_completion_tokens; most compatible servers want max_tokens.
    max_tokens_param: str = "max_tokens"
    needs_key: bool = True

    @property
    def api_key_env(self) -> str | None:
        return self.api_key_envs[0] if self.api_key_envs else None

    @property
    def destination(self) -> str:
        """Human-readable answer to "where does my prompt go?"."""
        if self.kind == "simulated":
            return "Nowhere (simulated in-process)"
        if self.kind == "anthropic":
            return "api.anthropic.com"
        host = urlparse(self.base_url or "").hostname or "unknown host"
        return f"{host} (this machine)" if self.local else host


def provider_presets() -> dict[str, ProviderPreset]:
    """Built fresh on each call so env changes (e.g. a new .env) are picked up."""
    ollama = ollama_base_url()
    custom_url = os.getenv("CUSTOM_OPENAI_BASE_URL")
    presets = [
        ProviderPreset("ollama", "Ollama", "openai_compat", f"{ollama}/v1", (), is_local_url(ollama), needs_key=False),
        ProviderPreset(
            "openai",
            "OpenAI",
            "openai_compat",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            ("OPENAI_API_KEY",),
            False,
            max_tokens_param="max_completion_tokens",
        ),
        ProviderPreset("anthropic", "Anthropic", "anthropic", None, ("ANTHROPIC_API_KEY",), False),
        ProviderPreset(
            "google",
            "Google Gemini",
            "openai_compat",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            False,
        ),
        ProviderPreset(
            "openrouter", "OpenRouter", "openai_compat", "https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",), False
        ),
        # LM Studio, vLLM, llama.cpp server, LocalAI... anything that speaks /v1/chat/completions.
        ProviderPreset(
            "custom",
            "Custom OpenAI-compatible",
            "openai_compat",
            custom_url,
            ("CUSTOM_OPENAI_API_KEY",),
            is_local_url(custom_url),
            needs_key=False,
        ),
        ProviderPreset("demo", "Demo (simulated)", "simulated", None, (), False, needs_key=False),
    ]
    return {p.id: p for p in presets}


def get_preset(provider_id: str) -> ProviderPreset:
    presets = provider_presets()
    if provider_id not in presets:
        raise KeyError(f"Unknown provider '{provider_id}'. Known: {', '.join(presets)}")
    return presets[provider_id]


def resolve_api_key(preset: ProviderPreset, overrides: dict[str, str] | None = None) -> str | None:
    """Session overrides (typed into the sidebar) win over environment variables."""
    overrides = overrides or {}
    for env in preset.api_key_envs:
        value = overrides.get(env) or os.getenv(env)
        if value:
            return value.strip()
    return None


def is_local_model(provider: str, model: str) -> bool:
    if provider == "demo":
        return model.startswith("demo-local")
    presets = provider_presets()
    return presets[provider].local if provider in presets else False
