import pytest

from slmbench.providers.simulated import SimulatedProvider


@pytest.fixture(autouse=True)
def fast_demo_and_isolated_runs(tmp_path, monkeypatch):
    """Demo models answer instantly, and saved runs go to a temp dir instead of runs/."""
    monkeypatch.setattr(SimulatedProvider, "time_scale", 0.0)
    monkeypatch.setenv("SLMBENCH_RUNS_DIR", str(tmp_path / "runs"))
    # Never let a developer's real keys leak into tests.
    for env in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"]:
        monkeypatch.delenv(env, raising=False)
