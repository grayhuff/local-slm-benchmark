"""Sidebar: model selection, prompt source, run settings, costs and keys.

Returns a single :class:`SidebarConfig` so the rest of the app never reads
widget state directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import streamlit as st

from slmbench import suites
from slmbench.config import PROJECT_ROOT, ollama_base_url, provider_presets
from slmbench.costs import EnergyConfig, PricingTable
from slmbench.providers import ollama, remote_availability
from slmbench.providers.base import estimate_tokens
from slmbench.providers.simulated import DEMO_PROFILES
from slmbench.schemas import DEFAULT_SYSTEM_PROMPT, GenerationParams, ModelSpec, PromptItem
from slmbench.scoring import available_scorers

MAX_MODELS = 4
EXAMPLE_JSONL = PROJECT_ROOT / "data" / "example_prompts.jsonl"
# Rough output length used only for the pre-run cost estimate.
ASSUMED_OUTPUT_TOKENS = 150


@dataclass
class SidebarConfig:
    demo: bool
    models: list[ModelSpec]
    items: list[PromptItem]
    params: GenerationParams
    repeats: int
    warmup: bool
    judge: ModelSpec | None
    energy: EnergyConfig
    tolerance: float
    save_runs: bool
    run_name: str
    run_clicked: bool = False
    problems: list[str] = field(default_factory=list)


@st.cache_data(ttl=15, show_spinner=False)
def cached_ollama_status(base_url: str) -> ollama.OllamaStatus:
    return ollama.get_status(base_url)


def _remote_label(spec: str, pricing: PricingTable) -> str:
    provider, _, model = spec.partition(":")
    preset = provider_presets().get(provider)
    price = pricing.lookup(provider, model)
    cost = f" · ${price.input_per_mtok:g} / ${price.output_per_mtok:g}" if price else " · no price"
    return f"{model} · {preset.name if preset else provider}{cost}"


def _models_section(pricing: PricingTable, demo: bool, api_keys: dict[str, str], problems: list[str]) -> list[ModelSpec]:
    st.subheader("1 · Models", divider="gray")
    if demo:
        picked = st.multiselect(
            "Simulated models",
            options=list(DEMO_PROFILES),
            default=list(DEMO_PROFILES),
            format_func=lambda m: f"{m} · {'on-device' if DEMO_PROFILES[m].local else 'API'}",
            help="Deterministic fake models that behave like a small/mid local model and a budget/frontier API.",
        )
        return [ModelSpec("demo", m, local=DEMO_PROFILES[m].local) for m in picked]

    status = cached_ollama_status(ollama_base_url())
    specs: list[ModelSpec] = []
    if status.running:
        names = [m.name for m in status.models]
        meta = {m.name: f"{m.parameter_size} {m.quantization}".strip() for m in status.models}
        local_picked = st.multiselect(
            f"Local · Ollama ({len(names)} pulled)",
            options=names,
            format_func=lambda n: f"{n} · {meta[n]}" if meta.get(n) else n,
            accept_new_options=True,
            placeholder="Choose or type a model tag",
            help="Runs on this machine via Ollama's OpenAI-compatible API. Prompts never leave the device.",
        )
        if not names:
            st.caption("No models pulled yet. Try `ollama pull llama3.2:3b` or `ollama pull qwen2.5:7b`.")
        specs += [ModelSpec("ollama", n, local=provider_presets()["ollama"].local) for n in local_picked]
    else:
        st.warning(
            f"**Ollama isn't reachable** at `{status.base_url}`.\n\n"
            "Install it from ollama.com, then run `ollama serve` and `ollama pull llama3.2:3b`.",
            icon=":material/power_off:",
        )
        if st.button("Check again", icon=":material/refresh:", width="stretch"):
            cached_ollama_status.clear()
            st.rerun()

    avail = {a.preset.id: a for a in remote_availability(api_keys)}
    options = [f"{p.provider}:{p.model}" for p in pricing.prices if p.provider in avail and avail[p.provider].usable]
    remote_picked = st.multiselect(
        "Remote · API",
        options=options,
        format_func=lambda s: _remote_label(s, pricing),
        accept_new_options=True,
        placeholder="Choose, or type provider:model",
        help="Prices are $ per 1M input / output tokens. Type e.g. `openrouter:meta-llama/llama-3.1-8b-instruct` "
        "or `custom:my-model` for anything not listed.",
    )
    missing = [a.preset.name for a in avail.values() if not a.usable]
    if missing:
        st.caption(f"No API key for {', '.join(missing)}. Add one in **API keys** below or in `.env`.")
    for raw in remote_picked:
        try:
            spec = ModelSpec.parse(raw)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if spec.provider not in provider_presets():
            problems.append(f"Unknown provider in '{raw}'. Use one of: {', '.join(provider_presets())}")
            continue
        specs.append(spec)
    return specs


def _prompts_section(problems: list[str]) -> list[PromptItem]:
    st.subheader("2 · Prompts", divider="gray")
    builtin = suites.load_builtin_suite()
    source = st.radio(
        "Prompt source",
        ["Built-in suite", "Custom prompt", "Upload JSONL", "Hugging Face sample"],
        captions=[f"{len(builtin)} graded questions", "one prompt, optional answer", "your own eval set", "e.g. GSM8K math"],
        label_visibility="collapsed",
    )
    if source == "Built-in suite":
        categories = sorted({i.category for i in builtin})
        chosen = st.multiselect("Categories", categories, default=categories, format_func=lambda c: c.replace("_", " "))
        items = [i for i in builtin if i.category in chosen]
    elif source == "Custom prompt":
        prompt = st.text_area("Prompt", placeholder="Summarize this support ticket in one sentence: …", height=120)
        expected = st.text_input("Expected answer (optional)", help="Leave blank to compare speed and cost only.")
        scorer = st.selectbox(
            "Scoring method", ["auto", *available_scorers()], help="'auto' picks from the shape of the expected answer."
        )
        items = [suites.custom_prompt_item(prompt, expected, None if scorer == "auto" else scorer)] if prompt.strip() else []
    elif source == "Upload JSONL":
        upload = st.file_uploader(
            "JSONL file", type=["jsonl", "json", "txt"], help='One JSON object per line with at least a "prompt" field.'
        )
        if EXAMPLE_JSONL.exists():
            st.download_button(
                "Example file",
                EXAMPLE_JSONL.read_text(),
                "example_prompts.jsonl",
                "application/jsonl",
                icon=":material/download:",
            )
        items = []
        if upload is not None:
            items, errors = suites.parse_jsonl(
                upload.getvalue().decode("utf-8", errors="replace"), source=upload.name.rsplit(".", 1)[0]
            )
            if errors:
                with st.expander(f"{len(errors)} line(s) skipped", icon=":material/warning:"):
                    st.code("\n".join(errors[:20]))
            st.caption(f"Loaded {len(items)} prompts.")
    else:
        name = st.selectbox("Dataset", list(suites.HF_DATASETS), format_func=lambda k: suites.HF_DATASETS[k]["label"])
        n = st.number_input("Sample size", 1, 50, 8)
        if st.button("Load sample", icon=":material/cloud_download:", width="stretch"):
            try:
                with st.spinner("Downloading from Hugging Face…"):
                    st.session_state.hf_items = suites.load_hf_sample(name, int(n))
            except Exception as exc:
                st.error(str(exc))
        items = st.session_state.get("hf_items", [])
        if items:
            st.caption(f"{len(items)} prompts loaded from {suites.HF_DATASETS[name]['path']}.")
    if not items:
        problems.append("Add at least one prompt.")
    return items


def _estimate_cost(
    models: list[ModelSpec], items: list[PromptItem], params: GenerationParams, repeats: int, pricing: PricingTable
) -> float | None:
    input_tokens = sum(estimate_tokens((params.system_prompt or "") + i.prompt) for i in items) * repeats
    output_tokens = ASSUMED_OUTPUT_TOKENS * len(items) * repeats
    total, unknown = 0.0, False
    for m in models:
        if m.local:
            continue
        c = pricing.request_cost(m.provider, m.model, input_tokens, output_tokens)
        if c is None:
            unknown = True
        else:
            total += c
    return None if unknown else total


def render_sidebar(pricing: PricingTable) -> SidebarConfig:
    problems: list[str] = []
    # Read key widgets up front: the model pickers render before the key inputs,
    # and should already reflect a key typed on the previous interaction.
    api_keys: dict[str, str] = {}
    for preset in provider_presets().values():
        if preset.api_key_env and st.session_state.get(f"key_{preset.api_key_env}"):
            api_keys[preset.api_key_env] = st.session_state[f"key_{preset.api_key_env}"].strip()
    st.session_state.api_keys = api_keys
    sb = st.sidebar
    with sb:
        st.title("Benchmark setup")
        demo = st.toggle(
            "Demo mode",
            key="demo_mode",
            help="Explore the full dashboard with simulated models. No Ollama or API keys needed. Results are clearly marked as simulated.",
        )
        models = _models_section(pricing, demo, api_keys, problems)
        if len(models) > MAX_MODELS:
            problems.append(f"Pick at most {MAX_MODELS} models ({len(models)} selected) so charts stay readable.")
        elif not models:
            problems.append("Pick at least one model.")

        items = _prompts_section(problems)

        st.subheader("3 · Settings", divider="gray")
        with st.expander("Generation", icon=":material/tune:"):
            temperature = st.slider(
                "Temperature",
                0.0,
                1.0,
                0.0,
                0.1,
                help="0 for repeatable runs. Reasoning models and Anthropic use their provider default.",
            )
            max_tokens = st.number_input(
                "Max output tokens",
                32,
                8192,
                1024,
                32,
                help="Reasoning models get at least 4096 so hidden thinking doesn't crowd out the answer.",
            )
            repeats = st.number_input("Repeats per prompt", 1, 5, 1, help="Run each prompt N times to smooth out latency noise.")
            system_prompt = st.text_area("System prompt", DEFAULT_SYSTEM_PROMPT, height=80)
            warmup = st.toggle(
                "Warm up local models first",
                True,
                help="Loads weights before timing so cold-start time isn't mixed into latency. Cold-start time is reported separately.",
            )
            timeout = st.slider("Request timeout (s)", 10, 600, 120, 10)
            judge_options = ["none"] + [m.key for m in models if m.provider != "demo"]
            judge_key = st.selectbox(
                "Judge model for open-ended answers",
                judge_options,
                format_func=lambda k: "None (use keyword fallback)" if k == "none" else k,
                disabled=demo,
                help="An LLM grades open-ended answers against a reference. It runs after the benchmark so it can't affect timings.",
            )
        with st.expander("Cost & scoring", icon=":material/payments:"):
            energy_defaults = EnergyConfig.from_env()
            watts = st.number_input(
                "Device power draw (W)",
                1.0,
                2000.0,
                energy_defaults.watts,
                5.0,
                help="Whole-system draw during inference. Rough guide: Apple M-series laptop 30–60 W, gaming desktop with RTX 4090 350–500 W.",
            )
            kwh = st.number_input("Electricity price ($/kWh)", 0.0, 2.0, energy_defaults.usd_per_kwh, 0.01, format="%.2f")
            include_energy = st.toggle("Include electricity in local cost", True)
            tolerance = st.slider(
                "'Good enough' accuracy tolerance (pts)",
                0,
                30,
                10,
                help="Best value = cheapest model within this many points of the top accuracy. Also used by the hybrid router.",
            )
        with st.expander("API keys", icon=":material/key:"):
            st.caption(
                "Keys typed here live only in this browser session's memory. They're never written to disk or saved with results."
            )
            for preset in provider_presets().values():
                if not preset.api_key_env or preset.local or preset.id == "custom":
                    continue
                env_set = bool(os.getenv(preset.api_key_env))
                st.text_input(
                    preset.name,
                    type="password",
                    key=f"key_{preset.api_key_env}",
                    placeholder="loaded from environment ✓" if env_set else preset.api_key_env,
                )
        with st.expander("History", icon=":material/history:"):
            save_runs = st.toggle("Save runs to runs/", True)
            run_name = st.text_input("Run name (optional)", placeholder="e.g. support-tickets v2")

        params = GenerationParams(
            temperature=temperature,
            max_tokens=int(max_tokens),
            system_prompt=system_prompt.strip() or None,
            timeout_s=float(timeout),
        )
        judge = next((m for m in models if m.key == judge_key), None)

        n_requests = len(models) * len(items) * int(repeats)
        estimate = _estimate_cost(models, items, params, int(repeats), pricing)
        st.divider()
        clicked = st.button(
            "Run benchmark", type="primary", icon=":material/play_arrow:", width="stretch", disabled=bool(problems)
        )
        if problems:
            st.caption(" · ".join(problems))
        else:
            est = (
                "unknown (unpriced model)" if estimate is None else f"≈ ${estimate:.4f}" if estimate < 1 else f"≈ ${estimate:.2f}"
            )
            st.caption(
                f"{len(models)} models × {len(items)} prompts × {repeats} = **{n_requests} requests** · est. API spend {est}"
            )

    return SidebarConfig(
        demo=demo,
        models=models,
        items=items,
        params=params,
        repeats=int(repeats),
        warmup=warmup,
        judge=judge,
        energy=EnergyConfig(watts=watts, usd_per_kwh=kwh, enabled=include_energy),
        tolerance=tolerance / 100,
        save_runs=save_runs,
        run_name=run_name.strip(),
        run_clicked=clicked,
        problems=problems,
    )
