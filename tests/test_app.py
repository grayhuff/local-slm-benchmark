"""Smoke-test the Streamlit app headlessly: it loads, and a demo run renders every tab."""

from streamlit.testing.v1 import AppTest

from slmbench.config import PROJECT_ROOT
from slmbench.schemas import DEFAULT_SYSTEM_PROMPT

APP = str(PROJECT_ROOT / "app.py")


def test_app_loads_with_onboarding():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Benchmark" in t.value for t in at.title)


def test_demo_run_renders_results():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["demo_mode"] = True
    at.run()
    next(b for b in at.button if b.label == "Run benchmark").click().run()
    assert not at.exception, at.exception
    result = at.session_state["result"]
    assert result is not None and result.simulated
    assert len(result.records) == 4 * 14
    assert result.config["system_prompt"] == DEFAULT_SYSTEM_PROMPT  # same default as the CLI
    markdown = " ".join(m.value for m in at.markdown)
    assert "Recommendation" in markdown
    assert "As a config you could ship" in markdown  # hybrid router tab rendered
