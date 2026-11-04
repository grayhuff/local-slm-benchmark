"""Anthropic Messages API provider (native SDK).

Anthropic does offer an OpenAI-compatible endpoint, but the native SDK reports
exact usage on every stream and maps errors precisely, which matters for a
tool whose whole point is accurate cost and latency numbers.
"""

from __future__ import annotations

import time

import anthropic

from slmbench.providers.base import Provider
from slmbench.schemas import Generation, GenerationParams


class AnthropicProvider(Provider):
    def __init__(self, preset, api_key=None):
        super().__init__(preset, api_key)
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=2)

    def generate(self, model: str, prompt: str, params: GenerationParams, *, reasoning: bool = False) -> Generation:
        kwargs = {
            "model": model,
            "max_tokens": params.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": params.timeout_s,
        }
        # The current Messages API doesn't take sampling parameters, so
        # temperature is left at the provider's default here.
        if params.system_prompt:
            kwargs["system"] = params.system_prompt

        start = time.perf_counter()
        first_token_at: float | None = None
        parts: list[str] = []
        try:
            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    if first_token_at is None and text:
                        first_token_at = time.perf_counter()
                    parts.append(text)
                final = stream.get_final_message()
        except Exception as exc:
            return Generation(latency_s=time.perf_counter() - start, error=self._friendly_error(exc, model))
        end = time.perf_counter()

        usage = final.usage
        input_tokens = (usage.input_tokens or 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        return Generation(
            text="".join(parts),
            ttft_s=(first_token_at - start) if first_token_at else None,
            latency_s=end - start,
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            finish_reason=final.stop_reason,
        )

    def _friendly_error(self, exc: Exception, model: str) -> str:
        if isinstance(exc, anthropic.APITimeoutError):
            return "Anthropic: request timed out"
        if isinstance(exc, anthropic.APIConnectionError):
            return "Anthropic: connection failed"
        if isinstance(exc, anthropic.AuthenticationError):
            return "Anthropic: authentication failed. Check ANTHROPIC_API_KEY."
        if isinstance(exc, anthropic.NotFoundError):
            return f"Anthropic: model '{model}' not found"
        if isinstance(exc, anthropic.RateLimitError):
            return "Anthropic: rate limited"
        if isinstance(exc, anthropic.APIStatusError):
            return f"Anthropic: HTTP {exc.status_code}: {str(exc)[:200]}"
        return f"Anthropic: {exc.__class__.__name__}: {str(exc)[:200]}"
