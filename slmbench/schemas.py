"""Plain dataclasses shared across the harness.

Keeping these dependency-free (no pandas, no SDK types) means every layer —
providers, scoring, analysis, UI — speaks the same small vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Shared by the dashboard and the CLI. Small models are sensitive to the system
# prompt, so two entry points with different defaults report different accuracy.
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."


@dataclass
class PromptItem:
    """One benchmark question.

    ``expected`` may be a string, number, list of acceptable answers, or a dict
    (for JSON-output tasks). ``scorer=None`` lets :mod:`slmbench.scoring` pick a
    sensible method from the shape of ``expected``.
    """

    id: str
    prompt: str
    category: str = "custom"
    expected: Any = None
    scorer: str | None = None
    scorer_args: dict[str, Any] = field(default_factory=dict)
    reference: str | None = None  # an ideal answer; used by judge/fuzzy scoring and demo mode
    distractor: str | None = None  # a plausible wrong answer; demo mode only


@dataclass(frozen=True)
class ModelSpec:
    """A model to benchmark, addressed as ``provider:model``."""

    provider: str
    model: str
    local: bool = False

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    @classmethod
    def parse(cls, spec: str, local: bool | None = None) -> ModelSpec:
        """Parse ``"ollama:llama3.2:3b"`` → provider ``ollama``, model ``llama3.2:3b``.

        Only the first colon separates provider from model, because Ollama tags
        contain colons themselves.
        """
        provider, sep, model = spec.partition(":")
        if not sep or not model:
            raise ValueError(f"Model spec must look like 'provider:model', got {spec!r}")
        if local is None:
            from slmbench.config import is_local_model

            local = is_local_model(provider, model)
        return cls(provider=provider.strip(), model=model.strip(), local=local)


@dataclass
class GenerationParams:
    temperature: float = 0.0
    max_tokens: int = 1024
    system_prompt: str | None = None
    timeout_s: float = 120.0


@dataclass
class Generation:
    """What a provider returns for a single request. Providers never raise;
    failures are reported through ``error`` so one bad request can't abort a run."""

    text: str = ""
    ttft_s: float | None = None
    latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_estimated: bool = False
    finish_reason: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ScoreResult:
    score: float | None  # 0..1, or None when the item has nothing to score against
    correct: bool | None
    method: str
    detail: str = ""

    @classmethod
    def unscored(cls, detail: str = "no expected answer") -> ScoreResult:
        return cls(score=None, correct=None, method="none", detail=detail)
