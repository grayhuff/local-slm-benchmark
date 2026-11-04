"""Main-area tabs. Each ``render_*`` function draws one tab from a
:class:`ViewContext`, which bundles the (re-priced) results so every tab
shows the same numbers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.theme import Palette
from slmbench import analysis, storage
from slmbench.config import PRICING_PATH, get_preset, ollama_base_url
from slmbench.costs import EnergyConfig, PricingTable
from slmbench.providers import ollama
from slmbench.runner import BenchmarkRunner, RunConfig, RunResult
from slmbench.schemas import ModelSpec
from slmbench.scoring import SCORER_DESCRIPTIONS
from slmbench.system import system_snapshot


@dataclass
class ViewContext:
    result: RunResult
    df: pd.DataFrame
    summary: pd.DataFrame
    pal: Palette
    colors: dict[str, str]
    names: dict[str, str]
    tolerance: float
    pricing: PricingTable
    energy: EnergyConfig

    @classmethod
    def build(cls, result: RunResult, pricing: PricingTable, energy: EnergyConfig, pal: Palette, tolerance: float) -> ViewContext:
        # Re-price on every render so edits to prices or electricity apply to past runs instantly.
        df = analysis.reprice(analysis.to_frame(result.records), pricing, energy)
        summary = analysis.summarize(df, result.model_info)
        order = list(summary["model_key"]) if not summary.empty else []
        names = dict(zip(summary["model_key"], summary["model"], strict=False)) if not summary.empty else {}
        return cls(result, df, summary, pal, pal.color_map(order), names, tolerance, pricing, energy)


SUMMARY_COLUMNS = {
    "model": st.column_config.TextColumn("Model"),
    "where": st.column_config.TextColumn("Runs on"),
    "accuracy": st.column_config.ProgressColumn(
        "Accuracy",
        min_value=0,
        max_value=100,
        format="%.0f%%",
        help="Mean score over scored prompts (0–1 per prompt, partial credit allowed)",
    ),
    "ttft_p50": st.column_config.NumberColumn("TTFT p50", format="%.2f s", help="Median time to first token"),
    "latency_p50": st.column_config.NumberColumn("Latency p50", format="%.2f s", help="Median end-to-end latency"),
    "latency_p95": st.column_config.NumberColumn("Latency p95", format="%.2f s"),
    "tokens_per_s": st.column_config.NumberColumn(
        "Tokens/s", format="%.0f", help="Decode speed after the first token, token-weighted across requests"
    ),
    "cost_per_1k": st.column_config.NumberColumn(
        "$ / 1k requests", format="$%.4f", help="API fees, or electricity for local models"
    ),
    "total_cost": st.column_config.NumberColumn("This run", format="$%.5f"),
    "input_tokens": st.column_config.NumberColumn("Input tok", format="%d"),
    "output_tokens": st.column_config.NumberColumn("Output tok", format="%d"),
    "errors": st.column_config.NumberColumn("Errors", format="%d"),
    "warmup_s": st.column_config.NumberColumn(
        "Cold start", format="%.1f s", help="One-off model load time before timing started (local only)"
    ),
    "memory_gb": st.column_config.NumberColumn("Memory", format="%.1f GB", help="Resident size reported by Ollama after the run"),
}


def _summary_view(ctx: ViewContext, compact: bool = False) -> pd.DataFrame:
    s = ctx.summary.copy()
    s["where"] = ["🔒 On-device" if local else "☁ API" for local in s["local"]]
    s["accuracy"] = s["accuracy"] * 100  # progress bars read better as 0–100
    cols = ["model", "where", "accuracy", "ttft_p50", "latency_p50", "tokens_per_s", "cost_per_1k"]
    if not compact:
        cols += ["latency_p95", "total_cost", "input_tokens", "output_tokens", "errors", "warmup_s", "memory_gb"]
    return s[cols]


# --------------------------------------------------------------------------- header & onboarding


def render_header(demo: bool, api_keys: dict[str, str]) -> None:
    from dashboard.sidebar import cached_ollama_status
    from slmbench.providers import remote_availability

    st.title("Local SLM Benchmark & Cost Dashboard")
    st.html(
        '<p class="hero-sub">Run the same prompts on open-weights models on your machine and on commercial APIs, then compare '
        "<b>speed</b>, <b>accuracy</b> and <b>cost</b> side by side. Local runs never leave the device.</p>"
    )
    status = cached_ollama_status(ollama_base_url())
    with st.container(horizontal=True, gap="small"):
        if demo:
            st.badge("Demo mode · simulated results", icon=":material/science:", color="orange")
        if status.running:
            st.badge(
                f"Ollama {status.version or ''} · {len(status.models)} models", icon=":material/check_circle:", color="green"
            )
        else:
            st.badge("Ollama offline", icon=":material/power_off:", color="gray")
        for avail in remote_availability(api_keys):
            if avail.usable:
                st.badge(avail.preset.name, icon=":material/key:", color="blue")
            else:
                st.badge(f"{avail.preset.name} · no key", color="gray")


def render_onboarding() -> None:
    st.write("")
    steps = [
        (
            "1",
            "Pick models",
            "Choose up to 4: local Ollama models (llama3.2, qwen2.5, phi…) and API models (OpenAI, Anthropic, Gemini).",
        ),
        (
            "2",
            "Choose prompts",
            "Use the built-in 14-question suite (reasoning, knowledge, instructions, extraction, open-ended), paste a prompt, or upload JSONL.",
        ),
        (
            "3",
            "Run & decide",
            "Get latency, tokens/sec, exact API cost, accuracy, a plain-English recommendation and a hybrid routing plan.",
        ),
    ]
    for col, (num, title, body) in zip(st.columns(3), steps, strict=False):
        with col.container(border=True, height="stretch"):
            st.html(f'<div class="step-num">Step {num}</div>')
            st.markdown(f"**{title}**")
            st.caption(body)
    st.info(
        "**No Ollama or API keys yet?** Switch on **Demo mode** in the sidebar and press **Run benchmark** to explore "
        "every chart with simulated models in about ten seconds.",
        icon=":material/lightbulb:",
    )


# --------------------------------------------------------------------------- running


def run_benchmark(cfg, pricing: PricingTable) -> RunResult | None:
    config = RunConfig(
        models=cfg.models,
        items=cfg.items,
        params=cfg.params,
        repeats=cfg.repeats,
        warmup_local=cfg.warmup,
        judge=cfg.judge,
        energy=cfg.energy,
        pricing=pricing,
        api_keys=st.session_state.get("api_keys", {}),
        name=cfg.run_name,
    )
    runner = BenchmarkRunner(config)
    result: RunResult | None = None
    rows: list[dict] = []
    with st.status(f"Running {runner.total_requests} requests…", expanded=True) as status:
        bar = st.progress(0.0, text="Starting…")
        live = st.empty()
        last_draw = 0.0
        for event in runner.run():
            if event.record:
                r = event.record
                rows.append(
                    {
                        "": "✗" if r["error"] else ("✓" if r["correct"] else ("·" if r["correct"] is None else "✗")),
                        "model": r["model"],
                        "prompt": r["prompt_id"],
                        "TTFT (s)": r["ttft_s"],
                        "latency (s)": r["latency_s"],
                        "out tok": r["output_tokens"],
                        "note": r["error"] or r["score_detail"],
                    }
                )
            bar.progress(min(event.completed / max(event.total, 1), 1.0), text=event.message)
            if rows and (time.monotonic() - last_draw > 0.3 or event.kind == "done"):
                live.dataframe(
                    pd.DataFrame(rows[-8:][::-1]),
                    hide_index=True,
                    placeholder="–",
                    column_config={
                        "TTFT (s)": st.column_config.NumberColumn(format="%.2f"),
                        "latency (s)": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                last_draw = time.monotonic()
            if event.kind == "done":
                result = event.result
        if result is None:
            status.update(label="Run did not finish", state="error")
            return None
        errors = sum(1 for r in result.records if r["error"])
        label = f"Done: {len(result.records)} requests in {result.duration_s:.1f}s"
        if errors:
            label += f" · {errors} failed"
        status.update(label=label, state="error" if errors == len(result.records) else "complete", expanded=False)

    result.config["system"] = system_snapshot()
    if cfg.save_runs:
        path = storage.save_run(result)
        st.toast(f"Saved to runs/{path.name}", icon=":material/save:")
    return result


# --------------------------------------------------------------------------- overview


def _run_caption(ctx: ViewContext) -> None:
    r = ctx.result
    cfg = r.config
    started = datetime.fromisoformat(r.started_at).astimezone().strftime("%b %d, %H:%M")
    bits = [
        f"Run `{r.run_id}`" + (f" · **{r.name}**" if r.name else ""),
        started,
        f"{len(cfg.get('models', []))} models × {cfg.get('n_prompts', '?')} prompts × {cfg.get('repeats', 1)}",
        f"{len(r.records)} requests in {r.duration_s:.1f}s",
    ]
    if cfg.get("judge"):
        bits.append(f"judge `{cfg['judge']}` ({analysis.fmt_usd(r.judge_cost_usd)})")
    st.caption(" · ".join(bits))
    if r.simulated:
        st.warning(
            "**Demo mode:** these results come from simulated models and are for exploring the dashboard only.",
            icon=":material/science:",
        )
    failed = ctx.df[~ctx.df["ok"]]
    if not failed.empty:
        with st.expander(
            f"{len(failed)} request(s) failed",
            icon=":material/error:",
            expanded=failed["model_key"].nunique() == ctx.df["model_key"].nunique(),
        ):
            for key, group in failed.groupby("model_key", sort=False):
                st.markdown(f"**{ctx.names.get(key, key)}** · {len(group)} failed · {group['error'].iloc[0]}")


def _winner_card(container, title: str, winner, pal_color: str | None) -> None:
    with container.container(border=True, height="stretch"):
        if winner is None:
            st.html(
                f'<div class="card-label">{title}</div><p class="card-model">n/a</p><div class="card-note">needs scored prompts</div>'
            )
            return
        st.html(
            f'<div class="card-label">{title}</div>'
            f'<p class="card-model"><span style="color:{pal_color}">●</span> {winner.model}</p>'
            f'<div class="card-value">{winner.value}</div>'
            f'<div class="card-note">{winner.note}</div>'
        )
        if winner.local:
            st.badge("On-device", icon=":material/lock:", color="green")
        else:
            st.badge("API", icon=":material/cloud:", color="blue")


def render_overview(ctx: ViewContext) -> None:
    _run_caption(ctx)
    if ctx.summary["latency_p50"].isna().all():
        st.error("Every request failed, so there's nothing to compare. Check the errors above.")
        return
    winners = analysis.pick_winners(ctx.summary, ctx.tolerance)
    cards = [("fastest", "Fastest"), ("cheapest", "Cheapest"), ("accurate", "Most accurate"), ("value", "Best value")]
    for col, (key, title) in zip(st.columns(4), cards, strict=False):
        w = winners.get(key)
        _winner_card(col, title, w, ctx.colors.get(w.model_key) if w else None)

    lines = analysis.recommendation(ctx.summary, ctx.tolerance)
    if lines:
        with st.container(border=True):
            st.markdown("##### :material/lightbulb: Recommendation")
            for line in lines:
                st.markdown(line)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Accuracy vs cost**")
        st.caption("Up and to the left is better. The dotted line links models nothing else beats on both.")
        if ctx.summary["accuracy"].notna().any():
            st.plotly_chart(charts.accuracy_vs_cost(ctx.summary, ctx.pal, ctx.colors), config={"displayModeBar": False})
        else:
            st.info("No scorable prompts in this run. Add expected answers to see accuracy.")
    with right:
        st.markdown("**Where the wait goes**")
        st.caption("Median time to first token (solid) plus generation time (light). Label = median total latency.")
        st.plotly_chart(charts.latency_breakdown(ctx.summary, ctx.pal, ctx.colors), config={"displayModeBar": False})

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Throughput**")
        st.caption("Output tokens per second once generation starts. Local speed depends on your hardware.")
        st.plotly_chart(charts.throughput(ctx.summary, ctx.pal, ctx.colors), config={"displayModeBar": False})
    with right:
        st.markdown("**Cost**")
        unit = st.segmented_control(
            "Cost unit", ["per 1k requests", "per request", "this run"], default="per 1k requests", label_visibility="collapsed"
        )
        st.plotly_chart(
            charts.cost(ctx.summary, ctx.pal, ctx.colors, unit or "per 1k requests"), config={"displayModeBar": False}
        )

    st.markdown("**At a glance**")
    st.dataframe(_summary_view(ctx, compact=True), hide_index=True, placeholder="–", column_config=SUMMARY_COLUMNS)


# --------------------------------------------------------------------------- results


def _mark(row) -> str:
    if not row["ok"]:
        return "⚠️ error"
    if pd.isna(row["score"]):
        return "– not scored"
    if row["score"] >= 1:
        return "✅ correct"
    if row["score"] <= 0:
        return "❌ wrong"
    return f"◐ {row['score'] * 100:.0f}%"


def render_results(ctx: ViewContext) -> None:
    st.markdown("**Model summary**")
    st.dataframe(_summary_view(ctx), hide_index=True, placeholder="–", column_config=SUMMARY_COLUMNS)

    run_json = json.dumps(ctx.result.to_dict(), indent=2, default=str)
    export = ctx.df.drop(columns=["ok"]).copy()
    with st.container(horizontal=True):
        st.download_button(
            "Per-request CSV",
            export.to_csv(index=False),
            f"slmbench_{ctx.result.run_id}_requests.csv",
            "text/csv",
            icon=":material/table:",
        )
        st.download_button(
            "Summary CSV",
            ctx.summary.to_csv(index=False),
            f"slmbench_{ctx.result.run_id}_summary.csv",
            "text/csv",
            icon=":material/summarize:",
        )
        st.download_button(
            "Full run JSON", run_json, f"slmbench_{ctx.result.run_id}.json", "application/json", icon=":material/data_object:"
        )

    st.markdown("**Every request**")
    f1, f2, f3 = st.columns([2, 2, 1.4])
    models = f1.multiselect("Models", list(ctx.names), format_func=ctx.names.get, placeholder="All models")
    categories = f2.multiselect("Categories", sorted(ctx.df["category"].unique()), placeholder="All categories")
    show = f3.segmented_control("Show", ["All", "Misses", "Errors"], default="All")
    view = ctx.df.copy()
    if models:
        view = view[view["model_key"].isin(models)]
    if categories:
        view = view[view["category"].isin(categories)]
    if show == "Misses":
        view = view[view["ok"] & (view["score"] < 1)]
    elif show == "Errors":
        view = view[~view["ok"]]
    rank = {k: i for i, k in enumerate(ctx.names)}
    view = view.assign(
        result=view.apply(_mark, axis=1), note=view["error"].fillna(view["score_detail"]), _r=view["model_key"].map(rank)
    )
    view = view.sort_values(["_r", "attempt"], kind="stable")
    st.dataframe(
        view[
            [
                "model",
                "prompt_id",
                "category",
                "result",
                "ttft_s",
                "latency_s",
                "tokens_per_s",
                "input_tokens",
                "output_tokens",
                "total_cost_usd",
                "scorer",
                "note",
                "response",
            ]
        ],
        hide_index=True,
        placeholder="–",
        column_config={
            "model": "Model",
            "prompt_id": "Prompt",
            "category": "Category",
            "result": "Result",
            "scorer": "Scorer",
            "ttft_s": st.column_config.NumberColumn("TTFT", format="%.2f s"),
            "latency_s": st.column_config.NumberColumn("Latency", format="%.2f s"),
            "tokens_per_s": st.column_config.NumberColumn("Tok/s", format="%.0f"),
            "input_tokens": st.column_config.NumberColumn("In", format="%d"),
            "output_tokens": st.column_config.NumberColumn("Out", format="%d"),
            "total_cost_usd": st.column_config.NumberColumn("Cost", format="$%.6f"),
            "note": st.column_config.TextColumn("Why", width="medium"),
            "response": st.column_config.TextColumn("Response", width="large"),
        },
    )

    st.markdown("**Compare answers side by side**")
    prompts = list(dict.fromkeys(ctx.df["prompt_id"]))
    pid = st.selectbox("Prompt", prompts, label_visibility="collapsed")
    rows = ctx.df[(ctx.df["prompt_id"] == pid) & (ctx.df["attempt"] == 0)]
    if not rows.empty:
        first = rows.iloc[0]
        st.code(first["prompt"], language=None, wrap_lines=True)
        if first["expected"]:
            st.caption(f"Expected: `{first['expected']}` · scored with `{first['scorer']}`")
        for col, (_, r) in zip(st.columns(len(rows)), rows.iterrows(), strict=False):
            with col.container(border=True, height="stretch"):
                st.html(f'<span style="color:{ctx.colors.get(r["model_key"])}">●</span> <b>{r["model"]}</b>')
                latency = f" · {r['latency_s']:.2f}s" if pd.notna(r["latency_s"]) else ""
                st.caption(f"{_mark(r)}{latency}")
                st.code(r["error"] or r["response"] or "(empty)", language=None, wrap_lines=True)
                if r["ok"] and r["score_detail"]:
                    st.caption(r["score_detail"])

    st.markdown("**Latency distribution**")
    st.caption("Each dot is one request. Medians hide slow outliers; this view doesn't.")
    st.plotly_chart(
        charts.latency_distribution(ctx.df, list(ctx.names), ctx.names, ctx.pal, ctx.colors), config={"displayModeBar": False}
    )


# --------------------------------------------------------------------------- accuracy


def render_accuracy(ctx: ViewContext) -> None:
    table = analysis.category_accuracy(ctx.df, ctx.result.model_info)
    if table.empty:
        st.info(
            "Nothing in this run was scorable. Give prompts an `expected` answer (or use the built-in suite) to measure accuracy."
        )
        return
    st.markdown("**Accuracy by task category**")
    st.caption(
        "Mean score per category. Small categories (2–4 prompts) swing a lot with each answer, so read them as directional."
    )
    st.plotly_chart(charts.category_heatmap(table, ctx.names, ctx.pal), config={"displayModeBar": False})

    st.markdown("**Prompt-by-prompt**")
    scored = ctx.df[ctx.df["attempt"] == 0].copy()
    scored["mark"] = scored.apply(_mark, axis=1)
    matrix = scored.pivot_table(index=["prompt_id", "category", "scorer"], columns="model_key", values="mark", aggfunc="first")
    matrix = matrix[[k for k in ctx.names if k in matrix.columns]].rename(columns=ctx.names).reset_index()
    st.dataframe(
        matrix,
        hide_index=True,
        placeholder="–",
        column_config={"prompt_id": "Prompt", "category": "Category", "scorer": "Scored by"},
    )

    with st.expander("How scoring works", icon=":material/rule:"):
        st.markdown(
            "Every prompt names a scoring method (or one is picked automatically from the shape of its expected answer). "
            "Each method returns a score from 0 to 1 and a one-line reason, shown in the **Results** tab. "
            "Model reasoning wrapped in `<think>…</think>` is stripped before scoring. "
            'To add a method, decorate a function with `@register("name")` in `slmbench/scoring.py`.'
        )
        st.dataframe(
            pd.DataFrame({"method": list(SCORER_DESCRIPTIONS), "what it checks": list(SCORER_DESCRIPTIONS.values())}),
            hide_index=True,
            placeholder="–",
        )


# --------------------------------------------------------------------------- hybrid router


def render_router(ctx: ViewContext) -> None:
    plan = analysis.hybrid_routing(ctx.df, ctx.tolerance)
    st.markdown(
        "**Should every request go to the same model?** For each task category this picks the *cheapest* model whose accuracy "
        f"is within **{ctx.tolerance * 100:.0f} pts** of the best model on that category (change it under *Cost & scoring*), "
        "then compares that mix with sending everything to the single most accurate model."
    )
    if plan is None:
        st.info(
            "The router needs at least two models scored on the same categories. Run the built-in suite with two or more models."
        )
        return
    acc_delta = round((plan.blended_accuracy - plan.baseline_accuracy) * 100)  # int, so never "-0"
    cost_delta = plan.blended_cost_per_1k - plan.baseline_cost_per_1k
    m = st.columns(4)
    m[0].metric(
        "Routed accuracy",
        analysis.fmt_pct(plan.blended_accuracy),
        (f"{acc_delta:+d} pts vs {plan.baseline_model}" if acc_delta else f"same as {plan.baseline_model}"),
        delta_color="normal" if acc_delta else "off",
        delta_arrow="auto" if acc_delta else "off",
        border=True,
    )
    m[1].metric(
        "Routed cost / 1k requests",
        analysis.fmt_usd(plan.blended_cost_per_1k),
        f"{'-' if cost_delta < 0 else '+'}{analysis.fmt_usd(abs(cost_delta))}",
        delta_color="inverse",
        border=True,
    )
    m[2].metric(
        "Cost saved",
        f"{plan.savings_pct * 100:.0f}%",
        f"vs always {plan.baseline_model}",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    m[3].metric(
        "Traffic kept on-device",
        f"{plan.local_share * 100:.0f}%",
        border=True,
        help="Share of prompts routed to local models, which never leave the machine",
    )

    table = plan.table.assign(
        route_to=[f"{'🔒' if loc else '☁'} {m}" for m, loc in zip(plan.table["route_to"], plan.table["on_device"], strict=False)],
        category=plan.table["category"].str.replace("_", " "),
        accuracy=plan.table["accuracy"] * 100,
        best_accuracy=plan.table["best_accuracy"] * 100,
    )
    st.dataframe(
        table[["category", "prompts", "route_to", "accuracy", "cost_per_1k", "best_model", "best_accuracy"]],
        hide_index=True,
        placeholder="–",
        column_config={
            "category": "Task category",
            "prompts": st.column_config.NumberColumn("Prompts", format="%d"),
            "route_to": "Route to",
            "accuracy": st.column_config.ProgressColumn("Its accuracy", min_value=0, max_value=100, format="%.0f%%"),
            "cost_per_1k": st.column_config.NumberColumn("$ / 1k", format="$%.4f"),
            "best_model": "Most accurate here",
            "best_accuracy": st.column_config.ProgressColumn("Best accuracy", min_value=0, max_value=100, format="%.0f%%"),
        },
    )
    for note in plan.notes:
        st.caption(f"⚠️ {note}")
    routes = {row.category: row.route_key for row in plan.table.itertuples()}
    st.markdown("**As a config you could ship**")
    st.code(
        "# Route by task type; classify incoming requests with a cheap rule or a small local model.\n"
        "ROUTES = "
        + json.dumps(routes, indent=4)
        + f'\nDEFAULT_ROUTE = "{plan.table.sort_values("accuracy").iloc[-1]["route_key"]}"',
        language="python",
    )
    st.caption(
        "In production the category isn't handed to you: route with a keyword rule, a classifier, or a small local model, "
        "and fall back to the API when the local answer fails validation (bad JSON, low confidence)."
    )


# --------------------------------------------------------------------------- history


def render_history(pricing: PricingTable, energy: EnergyConfig) -> None:
    entries = storage.list_runs()
    if not entries:
        st.info("No saved runs yet. Runs are saved to `runs/` automatically (toggle under *History* in the sidebar).")
        return
    index = pd.DataFrame(
        {
            "when": [
                datetime.fromisoformat(e.started_at).astimezone().strftime("%Y-%m-%d %H:%M") if e.started_at else ""
                for e in entries
            ],
            "name": [e.name or "" for e in entries],
            "models": [", ".join(m.split(":", 1)[1] for m in e.models) for e in entries],
            "prompts": [e.n_prompts for e in entries],
            "simulated": ["demo" if e.simulated else "" for e in entries],
            "file": [e.path.name for e in entries],
        }
    )
    st.markdown(f"**{len(entries)} saved run{'s' if len(entries) != 1 else ''}**")
    st.dataframe(index, hide_index=True, placeholder="–")
    labels = {e.path.name: f"{row.when} · {row.name or row.models}" for e, row in zip(entries, index.itertuples(), strict=False)}
    picked = st.multiselect(
        "Compare runs", list(labels), format_func=labels.get, max_selections=6, placeholder="Pick runs to compare side by side"
    )
    if picked:
        frames = []
        by_name = {e.path.name: e for e in entries}
        for name in picked:
            run = storage.load_run(by_name[name].path)
            df = analysis.reprice(analysis.to_frame(run.records), pricing, energy)
            frames.append(analysis.summarize(df, run.model_info).assign(run=labels[name]))
        combined = pd.concat(frames, ignore_index=True)
        combined["accuracy"] = combined["accuracy"] * 100
        st.dataframe(
            combined[["run", "model", "local", "accuracy", "ttft_p50", "latency_p50", "tokens_per_s", "cost_per_1k", "errors"]],
            hide_index=True,
            placeholder="–",
            column_config={k: v for k, v in SUMMARY_COLUMNS.items() if k in combined.columns}
            | {"local": st.column_config.CheckboxColumn("On-device")},
        )
        target = st.selectbox("Open one in the dashboard", picked, format_func=labels.get)
        if st.button("Open run", icon=":material/open_in_new:"):
            st.session_state.result = storage.load_run(by_name[target].path)
            st.rerun()


# --------------------------------------------------------------------------- pricing


def render_pricing(pricing: PricingTable) -> None:
    meta = pricing.meta
    st.markdown(
        f"API prices in USD per **1M tokens** (standard tier, no caching or batch discounts). Last verified "
        f"**{meta.get('last_verified', 'unknown')}**. Edits re-price every run instantly, including past ones, because costs are "
        "computed from stored token counts rather than saved as dollars."
    )
    sources = meta.get("sources", {})
    if sources:
        st.caption("Sources: " + " · ".join(f"[{k}]({v})" for k, v in sources.items()))
    edited = st.data_editor(
        pd.DataFrame(pricing.to_records()),
        num_rows="dynamic",
        hide_index=True,
        placeholder="–",
        key="pricing_editor",
        column_config={
            "provider": st.column_config.SelectboxColumn(
                "Provider", options=["openai", "anthropic", "google", "openrouter", "custom", "demo"], required=True
            ),
            "model": st.column_config.TextColumn("Model id", required=True),
            "input_per_mtok": st.column_config.NumberColumn("Input $/1M", format="$%.3f", min_value=0.0),
            "output_per_mtok": st.column_config.NumberColumn("Output $/1M", format="$%.3f", min_value=0.0),
            "reasoning": st.column_config.CheckboxColumn(
                "Reasoning", help="Bills hidden thinking tokens as output; temperature not supported"
            ),
            "notes": st.column_config.TextColumn("Notes", width="medium"),
        },
    )
    with st.container(horizontal=True):
        if st.button("Apply for this session", type="primary", icon=":material/check:"):
            st.session_state.pricing = PricingTable.from_records(edited.to_dict("records"), meta)
            st.rerun()
        if st.button("Save to config/pricing.json", icon=":material/save:"):
            table = PricingTable.from_records(
                edited.to_dict("records"), meta | {"last_verified": datetime.now().date().isoformat()}
            )
            table.save(PRICING_PATH)
            st.session_state.pricing = table
            st.toast("Pricing saved", icon=":material/save:")
            st.rerun()


# --------------------------------------------------------------------------- privacy & trade-offs


TRADEOFFS = """
#### When local wins
- **Privacy and compliance.** Prompts and documents never leave the machine: no vendor data-processing agreement, no retention questions. Often the deciding factor for healthcare, legal and internal data.
- **High, steady volume of narrow tasks.** Extraction, classification, routing, redaction and short summaries are where 3–8B models get close to API quality, and marginal cost is only electricity.
- **Latency to first token.** No network round trip or queue. On decent hardware small models often start answering before an API request has even reached the provider.
- **Offline or air-gapped.** Works on a plane, a factory floor or a locked-down network.

#### When the API is worth paying for
- **Hard reasoning, long context and broad knowledge.** Small models have a real quality ceiling. Multi-step math, code and nuanced writing are where the gap shows first.
- **Spiky or low volume.** If you send a few thousand requests a month, API costs are pennies and you skip running hardware.
- **Throughput at scale.** One laptop serves one request at a time well. Hundreds of concurrent users need serving infrastructure (vLLM, TGI), and that has its own cost.

#### Hardware rule of thumb (4-bit quantized)
| Model size | Memory needed | Typical hardware |
|---|---|---|
| 1–3B | 2–3 GB | any modern laptop, CPU is OK |
| 7–8B | 5–6 GB | 16 GB Apple Silicon, or a GPU with 8 GB+ VRAM |
| 13–14B | 9–10 GB | 32 GB Apple Silicon, or a GPU with 12–16 GB VRAM |
| 30–34B | 20–22 GB | 64 GB Apple Silicon, or a 24 GB GPU |

If a model doesn't fit in VRAM, Ollama splits it across CPU and GPU and speed drops sharply. Check the GPU share below.

#### When hybrid routing makes sense
Send the easy, high-volume categories to a local model and escalate the hard ones (or anything that fails validation, like malformed JSON) to an API. The **Hybrid router** tab estimates the savings from your own results. The usual outcome is most of the cost savings for a few points of accuracy.
"""


def render_privacy(models: list[ModelSpec], result: RunResult | None) -> None:
    keys = list(result.model_info) if result else [m.key for m in models]
    rows = []
    for key in keys:
        provider, _, model = key.partition(":")
        try:
            preset = get_preset(provider)
        except KeyError:
            continue
        local = result.model_info[key]["local"] if result else next((m.local for m in models if m.key == key), preset.local)
        rows.append(
            {
                "model": model,
                "runs on": "🔒 This machine" if local else f"☁ {preset.name}",
                "prompt is sent to": preset.destination if provider != "demo" else "Nowhere (simulated)",
                "leaves this machine": "No" if (local or provider == "demo") else "Yes",
            }
        )
    st.markdown("**Where your prompts go**")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, placeholder="–")
    else:
        st.caption("Select models in the sidebar to see where their data goes.")
    st.markdown(
        "- **Local models** are called at `localhost` through Ollama. Prompts, documents and answers stay in this machine's memory.\n"
        "- **API keys** typed in the sidebar live only in this session's memory. They're never written to disk or into saved runs.\n"
        "- **Saved runs** (prompts and responses included) are written only to the local `runs/` folder.\n"
        "- **Streamlit usage statistics are switched off** in `.streamlit/config.toml`, so the dashboard itself sends no telemetry."
    )

    st.markdown("**This machine**")
    snap = system_snapshot()
    gpus = snap.get("gpus") or [{"name": "none detected"}]
    vram = gpus[0].get("vram_total_gb")
    tiles = [
        ("CPU", snap.get("cpu") or "?", f"{snap.get('cpu_cores', '?')} cores"),
        ("RAM", f"{snap.get('ram_total_gb', '?')} GB", f"{snap.get('ram_available_gb', '?')} GB free"),
        ("GPU", gpus[0]["name"], f"{vram} GB VRAM" if vram else "shared memory" if "unified" in gpus[0]["name"] else ""),
        ("OS", snap.get("os", "?"), f"Python {snap.get('python', '?')}"),
    ]
    # Plain text tiles rather than st.metric: hardware names are long and metric values truncate.
    for col, (label, value, sub) in zip(st.columns(4), tiles, strict=True):
        with col.container(border=True, height="stretch"):
            st.html(f'<div class="card-label">{label}</div><p class="card-model">{value}</p><div class="card-note">{sub}</div>')

    loaded = ollama.loaded_models(ollama_base_url())
    if loaded:
        st.markdown("**Models loaded in Ollama right now**")
        st.dataframe(
            pd.DataFrame(
                {
                    "model": [m.get("name") for m in loaded],
                    "memory (GB)": [round((m.get("size") or 0) / 1e9, 2) for m in loaded],
                    "in VRAM (GB)": [round((m.get("size_vram") or 0) / 1e9, 2) for m in loaded],
                    "on GPU": [f"{(m.get('size_vram') or 0) / m['size'] * 100:.0f}%" if m.get("size") else "?" for m in loaded],
                    "unloads at": [str(m.get("expires_at", ""))[:19].replace("T", " ") for m in loaded],
                }
            ),
            hide_index=True,
            placeholder="–",
        )
    st.divider()
    st.markdown(TRADEOFFS)
