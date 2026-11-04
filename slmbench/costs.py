"""API pricing and local electricity cost.

Costs are always *derived* from stored token counts and latencies, never the
other way round. That lets the dashboard re-price an old run instantly when
the user edits a price or their electricity rate.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slmbench.config import PRICING_PATH

# A dated snapshot like "gpt-4o-2024-08-06" or "claude-haiku-4-5-20251001"
# should inherit its base model's price, but "gpt-5.3" must NOT silently
# inherit "gpt-5"'s — so only numeric suffixes count as the same model.
_SNAPSHOT_SUFFIX = re.compile(r"^-\d[\d-]*$")


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    model: str
    input_per_mtok: float
    output_per_mtok: float
    reasoning: bool = False
    notes: str = ""

    def cost(self, input_tokens: int | None, output_tokens: int | None) -> float:
        return ((input_tokens or 0) * self.input_per_mtok + (output_tokens or 0) * self.output_per_mtok) / 1_000_000


@dataclass
class PricingTable:
    prices: list[ModelPrice]
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path = PRICING_PATH) -> PricingTable:
        data = json.loads(Path(path).read_text())
        return cls.from_records(data.get("models", []), meta=data.get("_meta", {}))

    @classmethod
    def from_records(cls, records: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> PricingTable:
        prices = []
        for r in records:
            if not r.get("provider") or not r.get("model"):
                continue  # blank rows from the dashboard's editable table
            prices.append(
                ModelPrice(
                    provider=str(r["provider"]).strip(),
                    model=str(r["model"]).strip(),
                    input_per_mtok=float(r.get("input_per_mtok") or 0.0),
                    output_per_mtok=float(r.get("output_per_mtok") or 0.0),
                    reasoning=bool(r.get("reasoning") or False),
                    notes=str(r.get("notes") or ""),
                )
            )
        return cls(prices=prices, meta=dict(meta or {}))

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(p) for p in self.prices]

    def save(self, path: Path = PRICING_PATH) -> None:
        records = []
        for p in self.prices:
            r = {k: v for k, v in asdict(p).items() if v not in ("", False)}
            r.setdefault("input_per_mtok", 0.0)
            r.setdefault("output_per_mtok", 0.0)
            records.append(r)
        Path(path).write_text(json.dumps({"_meta": self.meta, "models": records}, indent=2) + "\n")

    def models_for(self, provider: str) -> list[ModelPrice]:
        return [p for p in self.prices if p.provider == provider]

    def lookup(self, provider: str, model: str) -> ModelPrice | None:
        """Exact match first, then the longest base name that ``model`` is a dated snapshot of."""
        candidates = self.models_for(provider)
        for p in candidates:
            if p.model == model:
                return p
        best: ModelPrice | None = None
        for p in candidates:
            if model.startswith(p.model) and _SNAPSHOT_SUFFIX.match(model[len(p.model) :]):
                if best is None or len(p.model) > len(best.model):
                    best = p
        return best

    def request_cost(self, provider: str, model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
        """Dollar cost of one request, or ``None`` if we have no price for this model."""
        price = self.lookup(provider, model)
        return price.cost(input_tokens, output_tokens) if price else None


def price_request(
    *,
    provider: str,
    model: str,
    local: bool,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_s: float | None,
    pricing: PricingTable,
    energy: EnergyConfig,
) -> dict[str, float | None]:
    """Cost columns for one request. Local models pay electricity, never API fees."""
    if local:
        wh, energy_usd = energy.energy(latency_s)
        return {"api_cost_usd": 0.0, "energy_wh": wh, "energy_cost_usd": energy_usd}
    api = pricing.request_cost(provider, model, input_tokens, output_tokens)
    return {"api_cost_usd": api, "energy_wh": 0.0, "energy_cost_usd": 0.0}


@dataclass
class EnergyConfig:
    """Rough electricity model for local inference: whole-system watts × wall time × rate.

    It deliberately over-attributes (the machine would draw idle power anyway),
    so treat it as an upper bound, not a meter reading.
    """

    watts: float = 60.0
    usd_per_kwh: float = 0.17
    enabled: bool = True

    def energy(self, seconds: float | None) -> tuple[float, float]:
        """Return ``(watt_hours, usd)`` for a request that took ``seconds`` of wall time."""
        if not self.enabled or not seconds:
            return 0.0, 0.0
        wh = self.watts * seconds / 3600.0
        return wh, wh / 1000.0 * self.usd_per_kwh
