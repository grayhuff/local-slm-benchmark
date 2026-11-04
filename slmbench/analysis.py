"""Turn raw per-request records into decisions.

Everything here is a pure function of a records DataFrame (plus pricing), so
the dashboard, the CLI and the tests all share the exact same numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from slmbench.costs import EnergyConfig, PricingTable, price_request

COST_COLUMNS = ["api_cost_usd", "energy_wh", "energy_cost_usd"]


# --------------------------------------------------------------------------- formatting


def fmt_usd(value: float | None) -> str:
    """Readable dollars across six orders of magnitude ($12.40, $0.031, $0.00042)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if value == 0:
        return "$0"
    if abs(value) >= 100:
        return f"${value:,.0f}"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    digits = max(2, 2 - math.floor(math.log10(abs(value))))  # 3 significant figures
    return f"${value:.{digits}f}"


def fmt_ratio(big: float, small: float) -> str:
    if small <= 0:
        return "∞×"
    ratio = big / small
    return f"{ratio:,.0f}×" if ratio >= 20 else f"{ratio:.1f}×"


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value * 100:.0f}%"


# --------------------------------------------------------------------------- frames


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    for col in ["ttft_s", "latency_s", "gen_time_s", "tokens_per_s", "score", *COST_COLUMNS]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ok"] = df["error"].isna()
    return df


def reprice(df: pd.DataFrame, pricing: PricingTable, energy: EnergyConfig) -> pd.DataFrame:
    """Recompute cost columns from tokens + latency under the *current* pricing and energy settings."""
    if df.empty:
        return df
    df = df.copy()
    rows = [
        price_request(
            provider=r.provider,
            model=r.model,
            local=bool(r.local),
            input_tokens=None if pd.isna(r.input_tokens) else int(r.input_tokens),
            output_tokens=None if pd.isna(r.output_tokens) else int(r.output_tokens),
            latency_s=None if pd.isna(r.latency_s) else float(r.latency_s),
            pricing=pricing,
            energy=energy,
        )
        for r in df.itertuples()
    ]
    costs = pd.DataFrame(rows, index=df.index)
    for col in COST_COLUMNS:
        df[col] = pd.to_numeric(costs[col], errors="coerce")
    df["total_cost_usd"] = df["api_cost_usd"] + df["energy_cost_usd"]
    return df


def _throughput(group: pd.DataFrame) -> float:
    """Token-weighted decode speed: Σ(output−1) / Σ(time after first token).

    Averaging per-request rates would let one 2-token answer with a 1 ms
    generation phase report thousands of tokens/sec.
    """
    g = group[group["ok"] & (group["output_tokens"] >= 2) & (group["gen_time_s"] > 0)]
    total_time = g["gen_time_s"].sum()
    return float((g["output_tokens"] - 1).sum() / total_time) if total_time > 0 else float("nan")


def model_order(df: pd.DataFrame, model_info: dict[str, Any] | None = None) -> list[str]:
    """Selection order (``model_info`` preserves it), then anything else in first-seen order."""
    seen = list(dict.fromkeys(df["model_key"])) if not df.empty else []
    preferred = [k for k in (model_info or {}) if k in seen]
    return preferred + [k for k in seen if k not in preferred]


def summarize(df: pd.DataFrame, model_info: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
    """One row per model with every headline metric, in selection order."""
    if df.empty:
        return pd.DataFrame()
    if "total_cost_usd" not in df:
        df = df.assign(total_cost_usd=df["api_cost_usd"] + df["energy_cost_usd"])
    model_info = model_info or {}
    rows = []
    for key in model_order(df, model_info):
        g = df[df["model_key"] == key]
        ok = g[g["ok"]]
        scored = ok[ok["score"].notna()]
        n_ok = len(ok)
        unpriced = (~g["local"]) & g["api_cost_usd"].isna() & g["ok"]
        total_cost = float(ok["total_cost_usd"].sum()) if not unpriced.any() else float("nan")
        info = model_info.get(key, {})
        memory = info.get("memory") or {}
        rows.append(
            {
                "model_key": key,
                "model": g["model"].iloc[0],
                "provider": g["provider"].iloc[0],
                "local": bool(g["local"].iloc[0]),
                "requests": len(g),
                "errors": int((~g["ok"]).sum()),
                "scored": len(scored),
                "accuracy": float(scored["score"].mean()) if len(scored) else float("nan"),
                "ttft_p50": float(ok["ttft_s"].median()) if n_ok else float("nan"),
                "latency_p50": float(ok["latency_s"].median()) if n_ok else float("nan"),
                "latency_p95": float(ok["latency_s"].quantile(0.95)) if n_ok else float("nan"),
                "gen_time_p50": float(ok["gen_time_s"].median()) if n_ok else float("nan"),
                "tokens_per_s": _throughput(g),
                "input_tokens": int(ok["input_tokens"].fillna(0).sum()),
                "output_tokens": int(ok["output_tokens"].fillna(0).sum()),
                "tokens_estimated": bool(g["tokens_estimated"].fillna(False).any()),
                "api_cost": float(ok["api_cost_usd"].sum()) if not unpriced.any() else float("nan"),
                "energy_cost": float(ok["energy_cost_usd"].sum()),
                "energy_wh": float(ok["energy_wh"].sum()),
                "total_cost": total_cost,
                "cost_per_request": total_cost / n_ok if n_ok else float("nan"),
                "cost_per_1k": total_cost / n_ok * 1000 if n_ok else float("nan"),
                "priced": not unpriced.any(),
                "warmup_s": info.get("warmup_s"),
                "memory_gb": memory.get("memory_gb"),
                "vram_gb": memory.get("vram_gb"),
            }
        )
    return pd.DataFrame(rows)


def category_accuracy(df: pd.DataFrame, model_info: dict[str, Any] | None = None) -> pd.DataFrame:
    """Models × categories matrix of mean score (NaN where nothing was scorable)."""
    scored = df[df["ok"] & df["score"].notna()]
    if scored.empty:
        return pd.DataFrame()
    table = scored.pivot_table(index="model_key", columns="category", values="score", aggfunc="mean")
    return table.loc[[k for k in model_order(df, model_info) if k in table.index]]


# --------------------------------------------------------------------------- winners & recommendation


@dataclass
class Winner:
    title: str
    model_key: str
    model: str
    local: bool
    value: str
    note: str = ""


def _valid(summary: pd.DataFrame, col: str) -> pd.DataFrame:
    return summary[summary[col].notna()]


def pick_winners(summary: pd.DataFrame, tolerance: float = 0.10) -> dict[str, Winner]:
    """Fastest, cheapest, most accurate, and best value.

    *Best value* = the cheapest model whose accuracy is within ``tolerance``
    (absolute, 0.10 = 10 points) of the most accurate model. It answers the
    question teams actually ask: "what's the cheapest thing that's good enough?"
    """
    winners: dict[str, Winner] = {}
    if summary.empty:
        return winners

    def make(title: str, row: pd.Series, value: str, note: str = "") -> Winner:
        return Winner(title, row["model_key"], row["model"], bool(row["local"]), value, note)

    fast = _valid(summary, "latency_p50")
    if not fast.empty:
        row = fast.sort_values(["latency_p50", "ttft_p50"]).iloc[0]
        winners["fastest"] = make("Fastest", row, f"{row['latency_p50']:.2f}s", f"median latency · TTFT {row['ttft_p50']:.2f}s")

    cheap = _valid(summary, "cost_per_1k")
    if not cheap.empty:
        row = cheap.sort_values(["cost_per_1k", "latency_p50"]).iloc[0]
        winners["cheapest"] = make("Cheapest", row, fmt_usd(row["cost_per_1k"]), "per 1k requests")

    acc = _valid(summary, "accuracy")
    if not acc.empty:
        row = acc.assign(_c=acc["cost_per_1k"].fillna(math.inf)).sort_values(["accuracy", "_c"], ascending=[False, True]).iloc[0]
        winners["accurate"] = make("Most accurate", row, fmt_pct(row["accuracy"]), f"{int(row['scored'])} scored prompts")

        best = acc["accuracy"].max()
        good = acc[(acc["accuracy"] >= best - tolerance - 1e-9) & acc["cost_per_1k"].notna()]
        if not good.empty:
            row = good.sort_values(["cost_per_1k", "latency_p50"]).iloc[0]
            gap = (best - row["accuracy"]) * 100
            note = "top accuracy" if gap < 0.5 else f"{gap:.0f} pts below best"
            winners["value"] = make(
                "Best value", row, fmt_usd(row["cost_per_1k"]) + " / 1k", f"{fmt_pct(row['accuracy'])} · {note}"
            )
    return winners


def _cost_phrase(cost_local: float, cost_remote: float) -> str:
    if cost_local <= 0:
        return f"**no API cost vs {fmt_usd(cost_remote)}** per 1k requests"
    return f"**{fmt_ratio(cost_remote, cost_local)} cheaper** ({fmt_usd(cost_local)} vs {fmt_usd(cost_remote)} per 1k requests)"


def _speed_phrase(local_row: pd.Series, remote_row: pd.Series) -> str:
    lt, rt = local_row["latency_p50"], remote_row["latency_p50"]
    if pd.isna(lt) or pd.isna(rt) or lt <= 0 or rt <= 0:
        return ""
    name = f"`{local_row['model']}`"
    if lt < rt:
        return f"On this hardware {name} also responded **{fmt_ratio(rt, lt)} faster** (median {lt:.2f}s vs {rt:.2f}s)."
    return f"On this hardware {name} was **{fmt_ratio(lt, rt)} slower** to respond (median {lt:.2f}s vs {rt:.2f}s)."


def recommendation(summary: pd.DataFrame, tolerance: float = 0.10) -> list[str]:
    """Plain-English takeaways, e.g. "local X is 40× cheaper and only 7 pts less accurate"."""
    if summary.empty:
        return []
    lines: list[str] = []
    local = summary[summary["local"] & summary["latency_p50"].notna()]
    remote = summary[~summary["local"] & summary["latency_p50"].notna()]
    has_acc = summary["accuracy"].notna().any()

    if not local.empty and not remote.empty:
        if has_acc:
            best_local = local.sort_values(["accuracy", "cost_per_1k"], ascending=[False, True]).iloc[0]
            best_remote = remote.sort_values(["accuracy", "cost_per_1k"], ascending=[False, True]).iloc[0]
        else:
            best_local = local.sort_values("latency_p50").iloc[0]
            best_remote = remote.sort_values("cost_per_1k").iloc[0]
        lname, rname = f"`{best_local['model']}`", f"`{best_remote['model']}`"
        cost_l, cost_r = best_local["cost_per_1k"], best_remote["cost_per_1k"]
        cost_known = not (pd.isna(cost_l) or pd.isna(cost_r))
        speed = _speed_phrase(best_local, best_remote)

        if has_acc and not pd.isna(best_local["accuracy"]) and not pd.isna(best_remote["accuracy"]):
            gap = (best_remote["accuracy"] - best_local["accuracy"]) * 100
            acc_l, acc_r = fmt_pct(best_local["accuracy"]), fmt_pct(best_remote["accuracy"])
            cost = _cost_phrase(cost_l, cost_r) if cost_known and cost_r > cost_l else ""
            if gap <= 0:
                lines.append(
                    f"**Local wins outright.** {lname} matched or beat {rname} on accuracy ({acc_l} vs {acc_r})"
                    + (f" and is {cost}" if cost else "")
                    + ", with every prompt staying on this machine. "
                    + speed
                )
            elif gap <= tolerance * 100:
                lines.append(
                    f"**Local is good enough here.** {lname} is only **{gap:.0f} pts less accurate** than {rname} "
                    f"({acc_l} vs {acc_r})"
                    + (f" while being {cost}" if cost else "")
                    + ", and your data never leaves the device. "
                    + speed
                )
            else:
                lines.append(
                    f"**The API earns its price on this task.** {rname} is **{gap:.0f} pts more accurate** than the best "
                    f"local model, {lname} ({acc_r} vs {acc_l})"
                    + (f", for {fmt_usd(cost_r - cost_l)} more per 1k requests" if cost_known else "")
                    + ". "
                    + speed
                    + " See the Hybrid router tab: a split by task type often keeps most of the savings."
                )
        else:
            cost = _cost_phrase(cost_l, cost_r) if cost_known and cost_r > cost_l else ""
            lines.append(
                f"No scorable answers in this prompt set, so this compares speed and cost only. {lname}"
                + (f" is {cost} compared with {rname}." if cost else f" vs {rname}.")
                + " "
                + speed
            )
    elif has_acc:
        acc = summary[summary["accuracy"].notna()].sort_values("accuracy", ascending=False)
        top = acc.iloc[0]
        lines.append(f"`{top['model']}` scored highest at {fmt_pct(top['accuracy'])}.")
        if local.empty:
            lines.append("Add a local Ollama model to see what on-device inference would save.")
        else:
            lines.append("Add an API model to see how far local quality is from the hosted ceiling.")

    if summary["tokens_estimated"].any():
        lines.append(
            "_Some providers didn't report token usage; those counts (and costs) are estimated at ~4 characters per token._"
        )
    if (~summary["priced"]).any():
        missing = ", ".join(f"`{m}`" for m in summary.loc[~summary["priced"], "model"])
        lines.append(f"_No price on file for {missing}. Add it in the Pricing tab to include it in cost comparisons._")
    return [line.strip() for line in lines]


# --------------------------------------------------------------------------- hybrid routing


@dataclass
class RoutingPlan:
    table: pd.DataFrame  # one row per category
    blended_accuracy: float
    blended_cost_per_1k: float
    baseline_model: str
    baseline_accuracy: float
    baseline_cost_per_1k: float
    local_share: float  # fraction of prompts routed to on-device models
    notes: list[str] = field(default_factory=list)

    @property
    def savings_pct(self) -> float:
        if not self.baseline_cost_per_1k:
            return 0.0
        return 1.0 - self.blended_cost_per_1k / self.baseline_cost_per_1k


def hybrid_routing(df: pd.DataFrame, tolerance: float = 0.10) -> RoutingPlan | None:
    """Per task category, route to the cheapest model within ``tolerance`` of that category's best.

    This is an *offline* routing policy learned from the benchmark; in
    production the category would come from a cheap classifier or the
    calling feature. Categories with very few prompts are noisy — the UI says so.
    """
    if df.empty or "total_cost_usd" not in df:
        return None
    scored = df[df["ok"] & df["score"].notna() & df["total_cost_usd"].notna()]
    if scored.empty or scored["model_key"].nunique() < 2:
        return None

    stats = (
        scored.groupby(["category", "model_key"])
        .agg(
            model=("model", "first"),
            local=("local", "first"),
            accuracy=("score", "mean"),
            cost_per_req=("total_cost_usd", "mean"),
            latency=("latency_s", "median"),
        )
        .reset_index()
    )
    prompts_per_cat = scored.groupby("category")["prompt_id"].nunique()
    # Only compare models that were scored on every category, so the baseline is apples-to-apples.
    complete = stats.groupby("model_key")["category"].nunique()
    complete = complete[complete == stats["category"].nunique()].index
    stats = stats[stats["model_key"].isin(complete)]
    if stats["model_key"].nunique() < 2:
        return None

    rows = []
    for cat, group in stats.groupby("category"):
        best = group.sort_values(["accuracy", "cost_per_req"], ascending=[False, True]).iloc[0]
        ok = group[group["accuracy"] >= best["accuracy"] - tolerance - 1e-9]
        pick = ok.sort_values(["cost_per_req", "latency"]).iloc[0]
        rows.append(
            {
                "category": cat,
                "prompts": int(prompts_per_cat[cat]),
                "route_to": pick["model"],
                "route_key": pick["model_key"],
                "on_device": bool(pick["local"]),
                "accuracy": pick["accuracy"],
                "cost_per_1k": pick["cost_per_req"] * 1000,
                "best_model": best["model"],
                "best_accuracy": best["accuracy"],
            }
        )
    table = pd.DataFrame(rows)
    weights = table["prompts"] / table["prompts"].sum()

    # Baseline: always send everything to the single most accurate model.
    per_model = stats.merge(prompts_per_cat.rename("n"), left_on="category", right_index=True)
    overall = per_model.assign(wa=per_model["accuracy"] * per_model["n"], wc=per_model["cost_per_req"] * per_model["n"])
    overall = overall.groupby(["model_key", "model"])[["wa", "wc", "n"]].sum().reset_index()
    overall["accuracy"] = overall["wa"] / overall["n"]
    overall["cost_per_1k"] = overall["wc"] / overall["n"] * 1000
    baseline = overall.sort_values(["accuracy", "cost_per_1k"], ascending=[False, True]).iloc[0]

    notes = []
    if (table["prompts"] < 5).any():
        notes.append(
            "Some categories have fewer than 5 prompts, so per-category accuracy is noisy. Treat this as a starting hypothesis."
        )
    return RoutingPlan(
        table=table,
        blended_accuracy=float((table["accuracy"] * weights).sum()),
        blended_cost_per_1k=float((table["cost_per_1k"] * weights).sum()),
        baseline_model=str(baseline["model"]),
        baseline_accuracy=float(baseline["accuracy"]),
        baseline_cost_per_1k=float(baseline["cost_per_1k"]),
        local_share=float(weights[table["on_device"]].sum()),
        notes=notes,
    )
