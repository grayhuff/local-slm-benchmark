"""OpenAI-compatible Chat Completions provider.

One implementation covers OpenAI, Ollama (``/v1``), Google Gemini's
compatibility endpoint, OpenRouter, LM Studio, vLLM and llama.cpp — the
reason the harness uses the ``openai`` client everywhere it can.
"""

from __future__ import annotations

import time
from typing import Any

import openai

from slmbench.providers import ollama
from slmbench.providers.base import Provider, estimate_tokens
from slmbench.schemas import Generation, GenerationParams

# Reasoning models spend part of max_tokens on hidden thinking; with a small
# budget they can return an empty answer, so give them this floor.
REASONING_TOKEN_FLOOR = 4096


class OpenAICompatProvider(Provider):
    def __init__(self, preset, api_key=None):
        super().__init__(preset, api_key)
        if not preset.base_url:
            raise ValueError(f"{preset.name}: no base URL configured (set CUSTOM_OPENAI_BASE_URL)")
        # Ollama and most local servers ignore the key but the client requires one.
        self.client = openai.OpenAI(base_url=preset.base_url, api_key=api_key or "not-needed", max_retries=2)

    @property
    def is_ollama(self) -> bool:
        return self.preset.id == "ollama"

    def _request_kwargs(self, model: str, prompt: str, params: GenerationParams, reasoning: bool) -> dict[str, Any]:
        messages = []
        if params.system_prompt:
            messages.append({"role": "system", "content": params.system_prompt})
        messages.append({"role": "user", "content": prompt})
        max_tokens = max(params.max_tokens, REASONING_TOKEN_FLOOR) if reasoning else params.max_tokens
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            # Ask for a final usage chunk so token counts are exact, not estimated.
            "stream_options": {"include_usage": True},
            self.preset.max_tokens_param: max_tokens,
        }
        if not reasoning:  # reasoning models reject any temperature other than the default
            kwargs["temperature"] = params.temperature
        return kwargs

    def _create(self, kwargs: dict[str, Any], timeout: float):
        """Create the stream, adapting once per parameter that a server rejects.

        Servers disagree on ``max_tokens`` vs ``max_completion_tokens``,
        ``temperature`` support and ``stream_options``; rather than hard-code a
        matrix we drop/rename the offending parameter and retry.
        """
        adapted: set[str] = set()
        while True:
            try:
                return self.client.chat.completions.create(**kwargs, timeout=timeout)
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                if "temperature" in msg and "temperature" in kwargs and "temperature" not in adapted:
                    kwargs.pop("temperature")
                    adapted.add("temperature")
                elif "stream_options" in msg and "stream_options" in kwargs and "stream_options" not in adapted:
                    kwargs.pop("stream_options")
                    adapted.add("stream_options")
                elif "max_tokens" in kwargs and "max_tokens" in msg and "max_tokens" not in adapted:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    adapted.add("max_tokens")
                elif "max_completion_tokens" in kwargs and "max_completion_tokens" in msg and "mct" not in adapted:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    adapted.add("mct")
                else:
                    raise

    def generate(self, model: str, prompt: str, params: GenerationParams, *, reasoning: bool = False) -> Generation:
        kwargs = self._request_kwargs(model, prompt, params, reasoning)
        start = time.perf_counter()
        first_token_at: float | None = None
        parts: list[str] = []
        usage = None
        finish_reason = None
        try:
            stream = self._create(kwargs, timeout=params.timeout_s)
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                piece = (delta.content or "") if delta else ""
                # Thinking models stream reasoning in a side field; it still counts as the first token.
                extra = (getattr(delta, "model_extra", None) or {}) if delta else {}
                thinking = extra.get("reasoning") or extra.get("reasoning_content") or ""
                if first_token_at is None and (piece or thinking):
                    first_token_at = time.perf_counter()
                if piece:
                    parts.append(piece)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        except Exception as exc:
            return Generation(latency_s=time.perf_counter() - start, error=self._friendly_error(exc, model))
        end = time.perf_counter()

        text = "".join(parts)
        gen = Generation(
            text=text,
            ttft_s=(first_token_at - start) if first_token_at else None,
            latency_s=end - start,
            finish_reason=finish_reason,
        )
        if usage is not None and usage.completion_tokens is not None:
            gen.input_tokens = usage.prompt_tokens
            gen.output_tokens = usage.completion_tokens
        else:
            gen.input_tokens = estimate_tokens((params.system_prompt or "") + prompt)
            gen.output_tokens = estimate_tokens(text)
            gen.tokens_estimated = True
        if not text and finish_reason == "length":
            gen.error = "Empty answer: the token budget ran out (raise Max output tokens)"
        return gen

    def _friendly_error(self, exc: Exception, model: str) -> str:
        name = self.preset.name
        if isinstance(exc, openai.APITimeoutError):
            return f"{name}: request timed out"
        if isinstance(exc, openai.APIConnectionError):
            if self.is_ollama:
                return f"Can't reach Ollama at {self.preset.base_url}. Is it running? Start it with `ollama serve`."
            return f"{name}: connection failed ({exc.__class__.__name__})"
        if isinstance(exc, openai.AuthenticationError):
            env = self.preset.api_key_env or "API key"
            return f"{name}: authentication failed. Check {env}."
        if isinstance(exc, openai.NotFoundError):
            if self.is_ollama:
                return f"Model '{model}' isn't pulled locally. Run `ollama pull {model}`."
            return f"{name}: model '{model}' not found or not available to your account"
        if isinstance(exc, openai.RateLimitError):
            return f"{name}: rate limited or out of quota"
        if isinstance(exc, openai.APIStatusError):
            return f"{name}: HTTP {exc.status_code}: {str(exc)[:200]}"
        return f"{name}: {exc.__class__.__name__}: {str(exc)[:200]}"

    # ------------------------------------------------------------------ local-only extras

    def warmup(self, model: str, params: GenerationParams) -> float | None:
        if not self.preset.local:
            return None
        start = time.perf_counter()
        warm = GenerationParams(temperature=0.0, max_tokens=1, timeout_s=max(params.timeout_s, 300))
        result = self.generate(model, "Hi", warm)
        return None if result.error else time.perf_counter() - start

    def memory_snapshot(self, model: str) -> dict[str, Any] | None:
        if not self.is_ollama:
            return None
        root = self.preset.base_url.removesuffix("/v1")
        for loaded in ollama.loaded_models(root):
            if loaded.get("name") == model or loaded.get("model") == model:
                size = loaded.get("size") or 0
                vram = loaded.get("size_vram") or 0
                return {
                    "memory_gb": round(size / 1e9, 2),
                    "vram_gb": round(vram / 1e9, 2),
                    "gpu_share": round(vram / size, 2) if size else None,
                }
        return None
