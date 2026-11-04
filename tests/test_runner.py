import threading

from slmbench import storage, suites
from slmbench.providers.base import Provider
from slmbench.runner import BenchmarkRunner, RunConfig
from slmbench.schemas import Generation, ModelSpec

DEMO = ["demo:demo-local-3b", "demo:demo-local-8b", "demo:demo-api-mini", "demo:demo-api-frontier"]


def run_all(config, **kw):
    events = list(BenchmarkRunner(config, **kw).run())
    return events, events[-1].result


def test_demo_run_end_to_end():
    items = suites.load_builtin_suite()
    events, result = run_all(RunConfig(models=[ModelSpec.parse(m) for m in DEMO], items=items))
    assert events[-1].kind == "done"
    assert len(result.records) == 4 * len(items)
    assert result.simulated
    assert all(r["error"] is None for r in result.records)
    local = result.model_info["demo:demo-local-3b"]
    assert local["warmup_s"] and local["memory"]["memory_gb"] > 0
    # local models never pay API fees; API models always have a price
    for r in result.records:
        assert (r["api_cost_usd"] == 0) if r["local"] else (r["api_cost_usd"] > 0)
    # progress is monotonic and ends at the total
    done = [e.completed for e in events if e.kind == "result"]
    assert done == sorted(done) and done[-1] == len(result.records)


def test_repeats_and_round_trip_save(tmp_path):
    items = suites.load_builtin_suite()[:2]
    _, result = run_all(RunConfig(models=[ModelSpec.parse("demo:demo-api-mini")], items=items, repeats=3))
    assert len(result.records) == 6
    path = storage.save_run(result, tmp_path)
    loaded = storage.load_run(path)
    assert loaded.records == result.records
    assert [e.run_id for e in storage.list_runs(tmp_path)] == [result.run_id]


def test_missing_key_is_reported_not_raised():
    items = suites.load_builtin_suite()[:3]
    _, result = run_all(RunConfig(models=[ModelSpec.parse("openai:gpt-4o-mini")], items=items))
    assert len(result.records) == 3
    assert all("OPENAI_API_KEY" in r["error"] for r in result.records)


class FlakyProvider(Provider):
    """Fails every call; counts how many requests actually reach it."""

    calls = 0

    def generate(self, model, prompt, params, *, reasoning=False):
        FlakyProvider.calls += 1
        return Generation(error="server down")


def test_stops_calling_a_broken_model_after_repeated_failures():
    from slmbench.config import get_preset

    FlakyProvider.calls = 0
    items = suites.load_builtin_suite()
    factory = lambda pid, keys: FlakyProvider(get_preset("ollama"))  # noqa: E731
    _, result = run_all(
        RunConfig(models=[ModelSpec("ollama", "ghost", local=True)], items=items, warmup_local=False), provider_factory=factory
    )
    assert FlakyProvider.calls == 3
    assert len(result.records) == len(items)
    assert "Skipped after 3 consecutive failures" in result.records[-1]["error"]


class TrackingProvider(Provider):
    """Records whether two local requests ever overlap in time."""

    active = 0
    max_active = 0
    lock = threading.Lock()

    def generate(self, model, prompt, params, *, reasoning=False):
        import time

        with self.lock:
            TrackingProvider.active += 1
            TrackingProvider.max_active = max(TrackingProvider.max_active, TrackingProvider.active)
        time.sleep(0.005)
        with self.lock:
            TrackingProvider.active -= 1
        return Generation(text="5", ttft_s=0.001, latency_s=0.005, input_tokens=5, output_tokens=1)


def test_local_models_never_run_concurrently():
    from slmbench.config import get_preset

    TrackingProvider.active = TrackingProvider.max_active = 0
    factory = lambda pid, keys: TrackingProvider(get_preset("ollama"))  # noqa: E731
    models = [ModelSpec("ollama", "a", local=True), ModelSpec("ollama", "b", local=True)]
    run_all(RunConfig(models=models, items=suites.load_builtin_suite(), warmup_local=False), provider_factory=factory)
    assert TrackingProvider.max_active == 1


def test_judge_runs_after_generation_and_rescores():
    from slmbench.config import get_preset

    class Judge(Provider):
        def generate(self, model, prompt, params, *, reasoning=False):
            return Generation(text="CORRECT\nok", ttft_s=0.01, latency_s=0.02, input_tokens=10, output_tokens=2)

    items = [i for i in suites.load_builtin_suite() if i.scorer == "judge"]
    demo = ModelSpec.parse("demo:demo-api-mini")
    judge = ModelSpec("openai", "gpt-4o-mini")

    def factory(pid, keys):
        from slmbench.providers import create_provider

        return Judge(get_preset("openai")) if pid == "openai" else create_provider(pid, keys)

    events, result = run_all(RunConfig(models=[demo], items=items, judge=judge), provider_factory=factory)
    assert any(e.kind == "judge" for e in events)
    assert all(r["scorer"] == "judge" and r["correct"] for r in result.records)
    assert result.judge_cost_usd > 0
