"""Benchmark orchestration.

Scheduling is the main design decision here. Requests are grouped into
*lanes* that run concurrently, but each lane is strictly sequential:

* **All local models share one lane.** Two local models generating at once
  would fight over the same GPU/CPU and corrupt each other's latency numbers.
  Running model-by-model also keeps exactly one model loaded, so Ollama never
  swaps weights mid-benchmark.
* **Each remote model gets its own lane.** API calls don't contend with each
  other (or with local inference) in any way that matters, so running them in
  parallel cuts wall-clock time without biasing results.

Lanes push results onto a queue; :meth:`BenchmarkRunner.run` drains it on the
caller's thread and yields progress events, which is what lets Streamlit
update the progress bar live (Streamlit elements must be touched from the
script thread).
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from slmbench import __version__, scoring
from slmbench.config import get_preset
from slmbench.costs import EnergyConfig, PricingTable, price_request
from slmbench.providers import create_provider
from slmbench.providers.base import Provider
from slmbench.schemas import Generation, GenerationParams, ModelSpec, PromptItem

ProviderFactory = Callable[[str, dict[str, str]], Provider]

# Stop hammering a model that is clearly broken (not pulled, bad key, server down).
MAX_CONSECUTIVE_FAILURES = 3
# Per-request tokens/sec is meaningless for 1–3 token answers; aggregates use token-weighted rates.
MIN_TOKENS_FOR_RATE = 8


@dataclass
class RunConfig:
    models: list[ModelSpec]
    items: list[PromptItem]
    params: GenerationParams = field(default_factory=GenerationParams)
    repeats: int = 1
    warmup_local: bool = True
    judge: ModelSpec | None = None
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    pricing: PricingTable | None = None
    api_keys: dict[str, str] = field(default_factory=dict, repr=False)
    name: str = ""

    def summary(self) -> dict[str, Any]:
        """Serializable description of the run, with secrets left out."""
        return {
            "models": [m.key for m in self.models],
            "n_prompts": len(self.items),
            "categories": sorted({i.category for i in self.items}),
            "repeats": self.repeats,
            "temperature": self.params.temperature,
            "max_tokens": self.params.max_tokens,
            "system_prompt": self.params.system_prompt,
            "warmup_local": self.warmup_local,
            "judge": self.judge.key if self.judge else None,
            "energy": asdict(self.energy),
        }


@dataclass
class RunResult:
    run_id: str
    started_at: str
    duration_s: float
    config: dict[str, Any]
    records: list[dict[str, Any]]
    model_info: dict[str, dict[str, Any]]
    judge_cost_usd: float = 0.0
    simulated: bool = False
    name: str = ""
    version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ProgressEvent:
    kind: str  # "status" | "warmup" | "result" | "judge" | "done"
    message: str
    completed: int
    total: int
    record: dict[str, Any] | None = None
    result: RunResult | None = None


def _expected_display(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class BenchmarkRunner:
    def __init__(self, config: RunConfig, provider_factory: ProviderFactory | None = None):
        if not config.models:
            raise ValueError("Select at least one model")
        if not config.items:
            raise ValueError("No prompts to run")
        self.config = config
        self.pricing = config.pricing or PricingTable.from_file()
        self._factory = provider_factory or create_provider
        self._cancel = threading.Event()
        self.run_id = uuid.uuid4().hex[:8]

    @property
    def total_requests(self) -> int:
        return len(self.config.models) * len(self.config.items) * self.config.repeats

    def cancel(self) -> None:
        self._cancel.set()

    def _lanes(self) -> list[list[ModelSpec]]:
        local = [m for m in self.config.models if m.local]
        lanes = [local] if local else []
        lanes += [[m] for m in self.config.models if not m.local]
        return lanes

    # ------------------------------------------------------------------ worker side

    def _lane(self, models: list[ModelSpec], out: queue.Queue) -> None:
        cfg = self.config
        try:
            for spec in models:
                if self._cancel.is_set():
                    return
                try:
                    provider = self._factory(spec.provider, cfg.api_keys)
                except Exception as exc:
                    out.put(("skip", spec, str(exc)))
                    continue
                if spec.local and cfg.warmup_local:
                    out.put(("status", spec, f"Loading {spec.model} into memory (warm-up, not timed)…"))
                    out.put(("warmup", spec, provider.warmup(spec.model, cfg.params)))
                price = self.pricing.lookup(spec.provider, spec.model)
                reasoning = bool(price and price.reasoning)
                failures, any_success, last_error = 0, False, ""
                for item in cfg.items:
                    for attempt in range(cfg.repeats):
                        if self._cancel.is_set():
                            return
                        if failures >= MAX_CONSECUTIVE_FAILURES and not any_success:
                            gen = Generation(error=f"Skipped after {failures} consecutive failures: {last_error}")
                        else:
                            gen = provider.generate_item(spec.model, item, cfg.params, attempt=attempt, reasoning=reasoning)
                            if gen.ok:
                                failures, any_success = 0, True
                            else:
                                failures += 1
                                last_error = gen.error
                        out.put(("gen", spec, item, attempt, gen))
                if spec.local:
                    out.put(("memory", spec, provider.memory_snapshot(spec.model)))
        except Exception as exc:  # defensive: the consumer must never wait forever
            out.put(("lane_error", None, f"{exc.__class__.__name__}: {exc}"))
        finally:
            out.put(("lane_done", None, None))

    # ------------------------------------------------------------------ consumer side

    def _record(self, spec: ModelSpec, item: PromptItem, attempt: int, gen: Generation) -> dict[str, Any]:
        cfg = self.config
        gen_time = gen.latency_s - gen.ttft_s if gen.ttft_s is not None else None
        rate = None
        if gen.ok and gen_time and gen_time > 0 and (gen.output_tokens or 0) >= MIN_TOKENS_FOR_RATE:
            rate = (gen.output_tokens - 1) / gen_time  # first token arrives at TTFT, the rest during gen_time
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "model_key": spec.key,
            "provider": spec.provider,
            "model": spec.model,
            "local": spec.local,
            "prompt_id": item.id,
            "category": item.category,
            "attempt": attempt,
            "prompt": item.prompt,
            "expected": _expected_display(item.expected),
            "response": gen.text,
            "error": gen.error,
            "finish_reason": gen.finish_reason,
            "ttft_s": gen.ttft_s,
            "latency_s": gen.latency_s if gen.ok else None,
            "gen_time_s": gen_time if gen.ok else None,
            "input_tokens": gen.input_tokens,
            "output_tokens": gen.output_tokens,
            "tokens_estimated": gen.tokens_estimated,
            "tokens_per_s": rate,
        }
        record.update(
            price_request(
                provider=spec.provider,
                model=spec.model,
                local=spec.local,
                input_tokens=gen.input_tokens,
                output_tokens=gen.output_tokens,
                latency_s=gen.latency_s if gen.ok else None,
                pricing=self.pricing,
                energy=cfg.energy,
            )
        )
        if gen.ok:
            result = scoring.score(gen.text, item, scoring.ScoringContext())  # judge runs in a later phase
        else:
            result = scoring.ScoreResult(None, None, scoring.resolve_method(item), "request failed")
        record.update(score=result.score, correct=result.correct, scorer=result.method, score_detail=result.detail)
        return record

    def _judge_context(self) -> tuple[scoring.ScoringContext | None, dict[str, float]]:
        spent = {"usd": 0.0}
        judge = self.config.judge
        if judge is None:
            return None, spent
        try:
            provider = self._factory(judge.provider, self.config.api_keys)
        except Exception:
            return None, spent  # scoring falls back to each item's non-judge method
        params = GenerationParams(temperature=0.0, max_tokens=200, timeout_s=self.config.params.timeout_s)
        price = self.pricing.lookup(judge.provider, judge.model)

        def ask(prompt: str) -> str:
            gen = provider.generate(judge.model, prompt, params, reasoning=bool(price and price.reasoning))
            if not gen.ok:
                raise RuntimeError(gen.error)
            if not judge.local:
                spent["usd"] += self.pricing.request_cost(judge.provider, judge.model, gen.input_tokens, gen.output_tokens) or 0.0
            return gen.text

        return scoring.ScoringContext(judge=ask), spent

    def run(self) -> Iterator[ProgressEvent]:
        cfg = self.config
        total = self.total_requests
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        t0 = time.perf_counter()
        records: list[dict[str, Any]] = []
        judge_queue: list[tuple[int, PromptItem]] = []
        items_by_id = {i.id: i for i in cfg.items}
        model_info: dict[str, dict[str, Any]] = {}
        for m in cfg.models:
            preset = get_preset(m.provider)
            model_info[m.key] = {
                "provider_name": preset.name,
                "local": m.local,
                "destination": preset.destination,
                "warmup_s": None,
                "memory": None,
            }

        out: queue.Queue = queue.Queue()
        lanes = self._lanes()
        executor = ThreadPoolExecutor(max_workers=len(lanes), thread_name_prefix="slmbench-lane")
        try:
            for lane in lanes:
                executor.submit(self._lane, lane, out)
            finished_lanes = 0
            while finished_lanes < len(lanes):
                try:
                    msg = out.get(timeout=0.25)  # short timeout keeps the caller interruptible
                except queue.Empty:
                    continue
                kind, spec = msg[0], msg[1]
                if kind == "lane_done":
                    finished_lanes += 1
                elif kind == "status":
                    yield ProgressEvent("status", msg[2], len(records), total)
                elif kind == "warmup":
                    model_info[spec.key]["warmup_s"] = msg[2]
                    label = f"{msg[2]:.1f}s" if msg[2] is not None else "failed"
                    yield ProgressEvent("warmup", f"{spec.model} warm-up: {label}", len(records), total)
                elif kind == "memory":
                    model_info[spec.key]["memory"] = msg[2]
                elif kind == "skip":
                    for item in cfg.items:
                        for attempt in range(cfg.repeats):
                            records.append(self._record(spec, item, attempt, Generation(error=msg[2])))
                    yield ProgressEvent("result", f"{spec.key}: {msg[2]}", len(records), total, record=records[-1])
                elif kind == "lane_error":
                    yield ProgressEvent("status", f"Internal error in a worker: {msg[2]}", len(records), total)
                elif kind == "gen":
                    _, spec, item, attempt, gen = msg
                    record = self._record(spec, item, attempt, gen)
                    records.append(record)
                    if gen.ok and cfg.judge and scoring.resolve_method(item) == "judge":
                        judge_queue.append((len(records) - 1, item))
                    status = "✗ error" if gen.error else f"{gen.latency_s:.2f}s"
                    yield ProgressEvent("result", f"{spec.model} · {item.id} · {status}", len(records), total, record=record)
        finally:
            self._cancel.set()
            executor.shutdown(wait=False, cancel_futures=True)

        # Judge scoring runs after generation so a local judge can't slow down (and bias) the models under test.
        ctx, spent = self._judge_context() if judge_queue else (None, {"usd": 0.0})
        for n, (idx, _item) in enumerate(judge_queue, start=1):
            rec = records[idx]
            result = scoring.score(rec["response"], items_by_id[rec["prompt_id"]], ctx)
            rec.update(score=result.score, correct=result.correct, scorer=result.method, score_detail=result.detail)
            yield ProgressEvent("judge", f"Judging {n}/{len(judge_queue)} with {cfg.judge.model}", len(records), total)

        result = RunResult(
            run_id=self.run_id,
            started_at=started_at,
            duration_s=round(time.perf_counter() - t0, 2),
            config=cfg.summary(),
            records=records,
            model_info=model_info,
            judge_cost_usd=spent["usd"],
            simulated=any(m.provider == "demo" for m in cfg.models),
            name=cfg.name,
        )
        yield ProgressEvent("done", "Benchmark complete", len(records), total, result=result)
