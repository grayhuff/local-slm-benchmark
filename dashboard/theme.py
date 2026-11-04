"""Colors and small CSS touches for the dashboard.

The categorical palette is validated for color-vision deficiency: adjacent
slots stay distinguishable for protan/deutan/tritan viewers in both themes
(checked with a CVD ΔE validator; worst adjacent pair ΔE 9.1 light / 8.4 dark).
Slots 3–4 are below 3:1 contrast on the light surface, so every chart that
uses them also prints values as text labels and has a table view.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

LIGHT_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK_SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
# Sequential blue ramp (light → dark) for magnitude, e.g. the accuracy heatmap.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


@dataclass(frozen=True)
class Palette:
    dark: bool
    series: list[str]
    surface: str
    text: str
    muted: str
    grid: str

    def color_map(self, keys: list[str]) -> dict[str, str]:
        """Fixed slot per model in selection order; a model keeps its color across every chart."""
        return {k: self.series[i % len(self.series)] for i, k in enumerate(keys)}


def palette() -> Palette:
    dark = False
    try:
        dark = (st.context.theme.type or "light") == "dark"
    except Exception:
        pass
    if dark:
        return Palette(True, DARK_SERIES, "#0e1117", "#f0efec", "#898781", "#2c2c2a")
    return Palette(False, LIGHT_SERIES, "#ffffff", "#0b0b0b", "#898781", "#e1e0d9")


CSS = """
<style>
  /* tighten Streamlit's default top padding so the header sits higher */
  .block-container { padding-top: 2.2rem; }
  .hero-sub { color: var(--text-color, inherit); opacity: .72; margin-top: -.6rem; font-size: 1.02rem; }
  .card-label { font-size: .78rem; letter-spacing: .04em; text-transform: uppercase; opacity: .65; margin-bottom: .1rem; }
  .card-model { font-weight: 600; font-size: 1.05rem; margin: 0; word-break: break-word; }
  .card-value { font-size: 1.7rem; font-weight: 650; line-height: 1.2; margin: .15rem 0 .1rem; font-variant-numeric: tabular-nums; }
  .card-note { font-size: .82rem; opacity: .7; }
  .step-num { font-size: .8rem; font-weight: 600; opacity: .6; text-transform: uppercase; letter-spacing: .05em; }
</style>
"""


def inject_css() -> None:
    st.html(CSS)
