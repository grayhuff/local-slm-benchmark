import math

import pandas as pd
import pytest

from slmbench import analysis


def rec(model, local, score, latency, ttft, out_tok, cost, category="cat", prompt="p1", error=None):
    return {
        "model_key": f"x:{model}",
        "provider": "x",
        "model": model,
        "local": local,
        "prompt_id": prompt,
        "category": category,
        "attempt": 0,
        "error": error,
        "ttft_s": ttft,
        "latency_s": latency,
        "gen_time_s": latency - ttft if latency is not None else None,
        "input_tokens": 50,
        "output_tokens": out_tok,
        "tokens_estimated": False,
        "tokens_per_s": None,
        "api_cost_usd": 0.0 if local else cost,
        "energy_wh": 0.0,
        "energy_cost_usd": cost if local else 0.0,
        "score": score,
    }


@pytest.fixture
def df():
    rows = []
    for i, cat in enumerate(["easy", "easy", "hard", "hard"]):
        p = f"p{i}"
        rows.append(rec("local-7b", True, 1.0 if cat == "easy" else 0.0, 0.5, 0.1, 41, 0.00001, cat, p))
        rows.append(rec("api-mini", False, 1.0, 1.0, 0.4, 61, 0.001, cat, p))
    return analysis.to_frame(rows).assign(total_cost_usd=lambda d: d["api_cost_usd"] + d["energy_cost_usd"])


def test_summary_metrics(df):
    s = analysis.summarize(df).set_index("model")
    assert s.loc["local-7b", "accuracy"] == 0.5
    assert s.loc["api-mini", "accuracy"] == 1.0
    assert s.loc["local-7b", "latency_p50"] == 0.5
    # token-weighted decode speed: (41-1) tokens over 0.4 s
    assert s.loc["local-7b", "tokens_per_s"] == pytest.approx(100)
    assert s.loc["api-mini", "cost_per_1k"] == pytest.approx(1.0)


def test_errors_excluded_from_latency_but_counted(df):
    bad = analysis.to_frame(
        [
            *df.drop(columns=["ok", "total_cost_usd"]).to_dict("records"),
            rec("api-mini", False, None, None, None, None, None, error="boom"),
        ]
    )
    bad["total_cost_usd"] = bad["api_cost_usd"].fillna(0) + bad["energy_cost_usd"]
    s = analysis.summarize(bad).set_index("model")
    assert s.loc["api-mini", "errors"] == 1
    assert s.loc["api-mini", "latency_p50"] == 1.0


def test_winners(df):
    s = analysis.summarize(df)
    w = analysis.pick_winners(s, tolerance=0.10)
    assert w["fastest"].model == "local-7b"
    assert w["cheapest"].model == "local-7b"
    assert w["accurate"].model == "api-mini"
    # local is 50 pts behind, outside a 10-pt tolerance → API is the best value
    assert w["value"].model == "api-mini"
    assert analysis.pick_winners(s, tolerance=0.6)["value"].model == "local-7b"


def test_recommendation_mentions_both_models(df):
    text = " ".join(analysis.recommendation(analysis.summarize(df), 0.10))
    assert "local-7b" in text and "api-mini" in text and "more accurate" in text


def test_hybrid_router_routes_easy_local_hard_remote(df):
    plan = analysis.hybrid_routing(df, tolerance=0.0)
    routes = dict(zip(plan.table["category"], plan.table["route_to"], strict=True))
    assert routes == {"easy": "local-7b", "hard": "api-mini"}
    assert plan.blended_accuracy == 1.0
    assert plan.local_share == 0.5
    assert plan.savings_pct == pytest.approx(0.495, abs=0.01)


def test_router_needs_two_models(df):
    assert analysis.hybrid_routing(df[df["model"] == "api-mini"]) is None


@pytest.mark.parametrize(
    ("value", "text"),
    [(0, "$0"), (12.345, "$12.35"), (0.0314, "$0.0314"), (0.000421, "$0.000421"), (float("nan"), "n/a"), (1234, "$1,234")],
)
def test_fmt_usd(value, text):
    assert analysis.fmt_usd(value) == text


def test_reprice_applies_new_prices():
    from slmbench.costs import EnergyConfig, PricingTable

    rows = [rec("gpt-4o-mini", False, 1.0, 1.0, 0.2, 1_000_000, None) | {"provider": "openai", "input_tokens": 0}]
    df = analysis.to_frame(rows)
    table = PricingTable.from_records(
        [{"provider": "openai", "model": "gpt-4o-mini", "input_per_mtok": 0.15, "output_per_mtok": 0.6}]
    )
    out = analysis.reprice(df, table, EnergyConfig())
    assert out["api_cost_usd"].iloc[0] == pytest.approx(0.6)
    assert not math.isnan(out["total_cost_usd"].iloc[0])
    assert isinstance(out, pd.DataFrame)
