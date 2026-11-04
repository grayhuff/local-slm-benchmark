"""Plotly figures. Each function takes the summary/records frames and a
:class:`~dashboard.theme.Palette` and returns a figure; no Streamlit calls here.

Conventions: one model = one fixed color everywhere; values printed on the
marks (so color is never the only carrier of meaning); one y-axis per chart.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import BLUE_RAMP, Palette
from slmbench.analysis import fmt_usd

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _base(fig: go.Figure, pal: Palette, height: int, **layout) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=24, t=8, b=8),
        font=dict(family=FONT, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font=dict(family=FONT)),
        showlegend=False,
        **layout,
    )
    fig.update_xaxes(gridcolor=pal.grid, zeroline=False, linecolor=pal.grid)
    fig.update_yaxes(gridcolor=pal.grid, zeroline=False)
    return fig


def _labels(summary: pd.DataFrame) -> list[str]:
    return [f"{m}  {'🔒' if local else '☁'}" for m, local in zip(summary["model"], summary["local"], strict=False)]


def _hbar(summary: pd.DataFrame, values: pd.Series, text: list[str], pal: Palette, colors: dict, hover: str, title: str):
    labels = _labels(summary)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=[colors[k] for k in summary["model_key"]], cornerradius=4),
            text=text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=hover + "<extra></extra>",
            customdata=summary["model"],
        )
    )
    _base(fig, pal, height=90 + 46 * len(summary), bargap=0.45)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title=dict(text=title, font=dict(size=12, color=pal.muted)), rangemode="tozero")
    # Leave room for the outside labels.
    vmax = pd.to_numeric(values, errors="coerce").max()
    if pd.notna(vmax) and vmax > 0:
        fig.update_xaxes(range=[0, vmax * 1.28])
    return fig


def latency_breakdown(summary: pd.DataFrame, pal: Palette, colors: dict) -> go.Figure:
    """Median TTFT (solid) + median generation time (light) = where the wait goes."""
    labels = _labels(summary)
    model_colors = [colors[k] for k in summary["model_key"]]
    ttft = summary["ttft_p50"].fillna(0)
    gen = (summary["latency_p50"] - summary["ttft_p50"]).clip(lower=0).fillna(0)
    fig = go.Figure()
    fig.add_bar(
        x=ttft,
        y=labels,
        orientation="h",
        name="Time to first token",
        marker=dict(color=model_colors, line=dict(color=pal.surface, width=2)),
        hovertemplate="%{y}<br>TTFT (median): %{x:.2f}s<extra></extra>",
    )
    fig.add_bar(
        x=gen,
        y=labels,
        orientation="h",
        name="Generation",
        marker=dict(color=model_colors, opacity=0.42, line=dict(color=pal.surface, width=2), cornerradius=4),
        text=[f"{v:.2f}s" for v in summary["latency_p50"].fillna(0)],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Generation (median): %{x:.2f}s<extra></extra>",
    )
    _base(fig, pal, height=90 + 46 * len(summary), barmode="stack", bargap=0.45)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(
        title=dict(text="seconds · solid = time to first token, light = generation", font=dict(size=12, color=pal.muted))
    )
    vmax = summary["latency_p50"].max()
    if pd.notna(vmax) and vmax > 0:
        fig.update_xaxes(range=[0, vmax * 1.25])
    return fig


def throughput(summary: pd.DataFrame, pal: Palette, colors: dict) -> go.Figure:
    values = summary["tokens_per_s"]
    text = ["n/a" if pd.isna(v) else f"{v:.0f} tok/s" for v in values]
    return _hbar(
        summary,
        values.fillna(0),
        text,
        pal,
        colors,
        "%{customdata}<br>%{x:.1f} tokens/sec",
        "output tokens per second (decode, token-weighted)",
    )


def cost(summary: pd.DataFrame, pal: Palette, colors: dict, unit: str = "per 1k requests") -> go.Figure:
    col = {"per 1k requests": "cost_per_1k", "per request": "cost_per_request", "this run": "total_cost"}[unit]
    values = summary[col]
    text = [fmt_usd(v) for v in values]
    return _hbar(
        summary,
        values.fillna(0),
        text,
        pal,
        colors,
        f"%{{customdata}}<br>%{{x:$.5f}} {unit}",
        f"USD {unit} (API fees + local electricity)",
    )


def _pareto(points: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated models: nothing else is both cheaper (or equal) and more accurate."""
    pts = points.sort_values(["x", "accuracy"], ascending=[True, False])
    frontier, best = [], -1.0
    for _, row in pts.iterrows():
        if row["accuracy"] > best + 1e-9:
            frontier.append(row)
            best = row["accuracy"]
    return pd.DataFrame(frontier)


def accuracy_vs_cost(summary: pd.DataFrame, pal: Palette, colors: dict) -> go.Figure:
    """Up-and-left is better. Log cost axis because local vs frontier spans ~1000×."""
    data = summary[summary["accuracy"].notna() & summary["cost_per_1k"].notna()].copy()
    fig = go.Figure()
    if data.empty:
        return _base(fig, pal, 340)
    positive = data.loc[data["cost_per_1k"] > 0, "cost_per_1k"]
    floor = positive.min() / 4 if not positive.empty else 1e-4
    data["x"] = data["cost_per_1k"].where(data["cost_per_1k"] > 0, floor)  # $0 can't sit on a log axis

    frontier = _pareto(data)
    if len(frontier) > 1:
        fig.add_scatter(
            x=frontier["x"],
            y=frontier["accuracy"] * 100,
            mode="lines",
            line=dict(color=pal.muted, width=2, dash="dot"),
            hoverinfo="skip",
            name="Pareto frontier",
        )
    for _, row in data.iterrows():
        fig.add_scatter(
            x=[row["x"]],
            y=[row["accuracy"] * 100],
            mode="markers+text",
            marker=dict(
                size=16,
                color=colors[row["model_key"]],
                symbol="circle" if row["local"] else "diamond",
                line=dict(color=pal.surface, width=2),
            ),
            text=[row["model"]],
            textposition="top center",
            textfont=dict(size=12),
            name=row["model"],
            cliponaxis=False,
            hovertemplate=(
                f"<b>{row['model']}</b> ({'on-device' if row['local'] else 'API'})<br>"
                f"Accuracy: {row['accuracy'] * 100:.0f}%<br>Cost: {fmt_usd(row['cost_per_1k'])} per 1k requests<extra></extra>"
            ),
        )
    lo, hi = math.log10(data["x"].min()), math.log10(data["x"].max())
    pad = max(0.35, (hi - lo) * 0.15)
    _base(fig, pal, 360)
    fig.update_xaxes(
        type="log",
        range=[lo - pad, hi + pad],
        dtick=1,
        tickprefix="$",
        title=dict(text="cost per 1k requests (USD, log scale)", font=dict(size=12, color=pal.muted)),
    )
    ymin = max(0, data["accuracy"].min() * 100 - 15)
    fig.update_yaxes(range=[ymin, 108], ticksuffix="%", title=dict(text="accuracy", font=dict(size=12, color=pal.muted)))
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        showarrow=False,
        align="left",
        text="↖ better: accurate and cheap",
        font=dict(size=12, color=pal.muted),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.01,
        showarrow=False,
        align="right",
        text="● on-device   ◆ API",
        font=dict(size=12, color=pal.muted),
    )
    return fig


def category_heatmap(table: pd.DataFrame, names: dict[str, str], pal: Palette) -> go.Figure:
    z = table.values * 100
    text = [["–" if pd.isna(v) else f"{v:.0f}%" for v in row] for row in z]
    steps = len(BLUE_RAMP) - 1
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[c.replace("_", " ") for c in table.columns],
            y=[names.get(k, k) for k in table.index],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorscale=[[i / steps, c] for i, c in enumerate(BLUE_RAMP)],
            zmin=0,
            zmax=100,
            xgap=2,
            ygap=2,
            colorbar=dict(ticksuffix="%", thickness=10, len=0.9),
            hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
        )
    )
    _base(fig, pal, height=110 + 48 * len(table))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    return fig


def latency_distribution(df: pd.DataFrame, order: list[str], names: dict[str, str], pal: Palette, colors: dict) -> go.Figure:
    """Every request as a dot, with a box for the spread: medians hide tail latency."""
    fig = go.Figure()
    ok = df[df["ok"]]
    for key in order:
        g = ok[ok["model_key"] == key]
        if g.empty:
            continue
        fig.add_box(
            y=g["latency_s"],
            name=names.get(key, key),
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
            marker=dict(color=colors[key], size=8, line=dict(color=pal.surface, width=1)),
            line=dict(color=colors[key], width=2),
            fillcolor="rgba(0,0,0,0)",
            customdata=g["prompt_id"],
            hovertemplate="%{customdata}<br>%{y:.2f}s<extra></extra>",
        )
    _base(fig, pal, 340)
    fig.update_yaxes(title=dict(text="latency per request (s)", font=dict(size=12, color=pal.muted)), rangemode="tozero")
    return fig
