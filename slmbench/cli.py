"""Headless runner for scripts and CI.

python -m slmbench.cli models
python -m slmbench.cli run -m ollama:llama3.2:3b -m openai:gpt-4o-mini
python -m slmbench.cli run --demo            # simulated models, no setup needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slmbench import analysis, storage, suites
from slmbench.config import load_env, ollama_base_url
from slmbench.costs import EnergyConfig, PricingTable
from slmbench.providers import ollama, remote_availability
from slmbench.providers.simulated import DEMO_PROFILES
from slmbench.runner import BenchmarkRunner, RunConfig
from slmbench.schemas import DEFAULT_SYSTEM_PROMPT, GenerationParams, ModelSpec
from slmbench.system import system_snapshot

DEMO_MODELS = ["demo:demo-local-3b", "demo:demo-local-8b", "demo:demo-api-mini", "demo:demo-api-frontier"]


def cmd_models(_: argparse.Namespace) -> int:
    status = ollama.get_status(ollama_base_url())
    if status.running:
        print(f"Ollama {status.version or ''} at {status.base_url}")
        for m in status.models:
            print(f"  ollama:{m.name:<28} {m.parameter_size:>6} {m.quantization:<8} {m.size_gb:>5.1f} GB")
        if not status.models:
            print("  (no models pulled: try `ollama pull llama3.2:3b`)")
    else:
        print(f"Ollama: {status.error}")
    pricing = PricingTable.from_file()
    for avail in remote_availability():
        flag = "key set" if avail.has_key else f"missing {avail.preset.api_key_env}"
        models = ", ".join(p.model for p in pricing.models_for(avail.preset.id)) or "(add models in config/pricing.json)"
        print(f"{avail.preset.name} [{flag}]: {models}")
    print("Demo: " + ", ".join(f"demo:{m}" for m in DEMO_PROFILES))
    return 0


def build_config(args: argparse.Namespace) -> RunConfig:
    specs = [ModelSpec.parse(s) for s in (DEMO_MODELS if args.demo else args.model or [])]
    items = suites.load_jsonl_file(Path(args.prompts)) if args.prompts else suites.load_builtin_suite()
    energy = EnergyConfig.from_env()
    if args.watts is not None:
        energy.watts = args.watts
    if args.kwh_price is not None:
        energy.usd_per_kwh = args.kwh_price
    return RunConfig(
        models=specs,
        items=items,
        params=GenerationParams(temperature=args.temperature, max_tokens=args.max_tokens, system_prompt=args.system or None),
        repeats=args.repeats,
        energy=energy,
        judge=ModelSpec.parse(args.judge) if args.judge else None,
    )


def cmd_run(args: argparse.Namespace) -> int:
    config = build_config(args)
    if not config.models:
        print("Pass at least one --model provider:model (or --demo)", file=sys.stderr)
        return 2
    runner = BenchmarkRunner(config)
    result = None
    for event in runner.run():
        if event.kind in ("result", "warmup", "status", "judge"):
            print(f"[{event.completed:>3}/{event.total}] {event.message}", flush=True)
        if event.kind == "done":
            result = event.result
    assert result is not None
    result.config["system"] = system_snapshot()
    df = analysis.reprice(analysis.to_frame(result.records), runner.pricing, config.energy)
    summary = analysis.summarize(df, result.model_info)
    cols = ["model", "local", "accuracy", "ttft_p50", "latency_p50", "tokens_per_s", "cost_per_1k", "errors"]
    print()
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    for line in analysis.recommendation(summary):
        print("- " + line.replace("**", "").replace("`", ""))
    if not args.no_save:
        print(f"\nSaved {storage.save_run(result)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slmbench", description="Local SLM vs API benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list available local and API models").set_defaults(fn=cmd_models)
    run = sub.add_parser("run", help="run a benchmark")
    run.add_argument("-m", "--model", action="append", help="provider:model, repeatable")
    run.add_argument("--demo", action="store_true", help="use the four simulated demo models")
    run.add_argument("--prompts", help="JSONL file (default: built-in suite)")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help='system prompt (default: "%(default)s"; "" for none)')
    run.add_argument("--judge", default=None, help="provider:model used to grade open-ended answers")
    run.add_argument(
        "--watts", type=float, help="device power draw during local inference (default: $SLMBENCH_DEVICE_WATTS or 60)"
    )
    run.add_argument("--kwh-price", type=float, help="electricity price in USD per kWh (default: $SLMBENCH_USD_PER_KWH or 0.17)")
    run.add_argument("--no-save", action="store_true")
    run.set_defaults(fn=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
