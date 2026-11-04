"""Provider interface.

A provider turns ``(model, prompt, params)`` into a :class:`Generation` with
timing and token counts. Implementations must never raise from
:meth:`Provider.generate` — errors go in ``Generation.error`` — so a single
missing model or rate-limit can't abort a 60-request benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from slmbench.config import ProviderPreset
from slmbench.schemas import Generation, GenerationParams, PromptItem


def estimate_tokens(text: str) -> int:
    """~4 characters per token for English. Only used when a server omits usage."""
    return max(1, round(len(text) / 4)) if text else 0


class Provider(ABC):
    def __init__(self, preset: ProviderPreset, api_key: str | None = None):
        self.preset = preset
        self.api_key = api_key

    @abstractmethod
    def generate(self, model: str, prompt: str, params: GenerationParams, *, reasoning: bool = False) -> Generation:
        """Run one streamed chat completion and measure it."""

    def generate_item(
        self, model: str, item: PromptItem, params: GenerationParams, *, attempt: int = 0, reasoning: bool = False
    ) -> Generation:
        """Entry point used by the runner. Real providers only need the prompt text."""
        return self.generate(model, item.prompt, params, reasoning=reasoning)

    def warmup(self, model: str, params: GenerationParams) -> float | None:
        """Load the model before timing starts; returns seconds taken (cold-start time).

        Only meaningful for local servers, where the first request pays for
        loading weights into memory. Remote providers return ``None``.
        """
        return None

    def memory_snapshot(self, model: str) -> dict[str, Any] | None:
        """Resident memory of a loaded local model, if the server reports it."""
        return None
