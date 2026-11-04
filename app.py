"""Local SLM Benchmark & Cost Dashboard — Streamlit entry point.

streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from dashboard import views
from dashboard.sidebar import cached_ollama_status, render_sidebar
from dashboard.theme import inject_css, palette
from slmbench.config import load_env, ollama_base_url
from slmbench.costs import PricingTable
from slmbench.providers import remote_availability

st.set_page_config(page_title="Local SLM Benchmark", page_icon=":material/speed:", layout="wide")
load_env()
inject_css()


def init_state() -> None:
    if "pricing" not in st.session_state:
        try:
            st.session_state.pricing = PricingTable.from_file()
        except Exception as exc:
            st.error(f"Couldn't read config/pricing.json ({exc}). Costs will show as n/a until it's fixed.")
            st.session_state.pricing = PricingTable(prices=[])
    if "demo_mode" not in st.session_state:
        # First visit with nothing set up: start in demo mode so "Run" works immediately.
        nothing_configured = not cached_ollama_status(ollama_base_url()).running and not any(
            a.usable for a in remote_availability()
        )
        st.session_state.demo_mode = nothing_configured
    st.session_state.setdefault("result", None)


init_state()
pricing: PricingTable = st.session_state.pricing
cfg = render_sidebar(pricing)
views.render_header(cfg.demo, st.session_state.get("api_keys", {}))

if cfg.run_clicked:
    result = views.run_benchmark(cfg, pricing)
    if result is not None:
        st.session_state.result = result

result = st.session_state.result
tabs = st.tabs(
    [
        ":material/dashboard: Overview",
        ":material/table_rows: Results",
        ":material/target: Accuracy",
        ":material/alt_route: Hybrid router",
        ":material/history: History",
        ":material/sell: Pricing",
        ":material/shield_lock: Privacy & trade-offs",
    ]
)
ctx = views.ViewContext.build(result, pricing, cfg.energy, palette(), cfg.tolerance) if result else None

with tabs[0]:
    if ctx:
        views.render_overview(ctx)
    else:
        views.render_onboarding()
for tab, render in zip(tabs[1:4], [views.render_results, views.render_accuracy, views.render_router], strict=False):
    with tab:
        if ctx:
            render(ctx)
        else:
            st.info("Run a benchmark to see this view.", icon=":material/play_arrow:")
with tabs[4]:
    views.render_history(pricing, cfg.energy)
with tabs[5]:
    views.render_pricing(pricing)
with tabs[6]:
    views.render_privacy(cfg.models, result)
