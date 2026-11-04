"""Demo-mode provider: deterministic, clearly-labelled simulated models.

It lets anyone click "Run" and explore the full dashboard in seconds without
Ollama or API keys. Outputs are seeded from (model, prompt) so a demo run is
reproducible. Nothing here pretends to be a real model: names are generic
and every simulated run is flagged in the UI and in saved results.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass

from slmbench.providers.base import Provider, estimate_tokens
from slmbench.schemas import Generation, GenerationParams, PromptItem


@dataclass(frozen=True)
class DemoProfile:
    local: bool
    ttft_range: tuple[float, float]
    tps_range: tuple[float, float]
    skill: float  # base probability of a correct answer
    blurb: str


DEMO_PROFILES: dict[str, DemoProfile] = {
    "demo-local-3b": DemoProfile(True, (0.06, 0.20), (60, 80), 0.62, "small on-device model (~3B, 4-bit)"),
    "demo-local-8b": DemoProfile(True, (0.15, 0.40), (28, 38), 0.80, "mid-size on-device model (~8B, 4-bit)"),
    "demo-api-mini": DemoProfile(False, (0.35, 0.90), (70, 110), 0.84, "budget hosted API model"),
    "demo-api-frontier": DemoProfile(False, (0.55, 1.40), (45, 70), 0.96, "frontier hosted API model"),
}

# Harder categories pull weaker models down more than strong ones.
_DIFFICULTY = {"reasoning": 0.35, "instruction": 0.15, "extraction": 0.05, "knowledge": -0.10, "open_ended": 0.0}
_RAMBLE = ["Hope that helps!", "Let me know if you need more detail."]


def _rng(*parts: object) -> random.Random:
    seed = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]
    return random.Random(int(seed, 16))


class SimulatedProvider(Provider):
    # Fraction of the simulated latency we actually sleep, so progress bars move
    # realistically without making the demo slow. Tests set this to 0.
    time_scale: float = 0.15

    def generate(self, model: str, prompt: str, params: GenerationParams, *, reasoning: bool = False) -> Generation:
        return self._simulate(model, PromptItem(id="adhoc", prompt=prompt), params, attempt=0)

    def generate_item(self, model, item, params, *, attempt=0, reasoning=False) -> Generation:
        # Demo models need the item's reference/distractor to produce gradeable answers.
        return self._simulate(model, item, params, attempt)

    def _simulate(self, model: str, item: PromptItem, params: GenerationParams, attempt: int) -> Generation:
        profile = DEMO_PROFILES.get(model)
        if profile is None:
            return Generation(error=f"Unknown demo model '{model}'. Options: {', '.join(DEMO_PROFILES)}")
        rng = _rng(model, item.prompt, attempt)

        # Item-response style: every prompt has one shared difficulty, so stronger
        # models mostly solve a superset of what weaker ones solve (plus a little noise).
        difficulty = _rng("difficulty", item.prompt).random() * 0.9  # every prompt is solvable by a strong model
        p_correct = profile.skill - _DIFFICULTY.get(item.category, 0.0) * (1.0 - profile.skill)
        if item.reference is None:
            text = f"(Simulated {model} response) A concise answer to: {item.prompt[:80]}"
        elif difficulty + rng.uniform(-0.12, 0.12) < p_correct:
            text = item.reference
        else:
            text = item.distractor or "I'm not sure."
        if profile.local and rng.random() < 0.3:  # small models pad answers a bit more
            text = f"{text} {rng.choice(_RAMBLE)}"

        output_tokens = estimate_tokens(text) + rng.randint(0, 6)
        ttft = rng.uniform(*profile.ttft_range) + estimate_tokens(item.prompt) / 4000
        tps = rng.uniform(*profile.tps_range)
        latency = ttft + max(output_tokens - 1, 0) / tps
        if self.time_scale:
            time.sleep(min(latency * self.time_scale, 0.5))
        return Generation(
            text=text,
            ttft_s=ttft,
            latency_s=latency,
            input_tokens=estimate_tokens((params.system_prompt or "") + item.prompt) + 8,
            output_tokens=output_tokens,
            finish_reason="stop",
        )

    def warmup(self, model: str, params: GenerationParams) -> float | None:
        profile = DEMO_PROFILES.get(model)
        if not profile or not profile.local:
            return None
        seconds = _rng("warmup", model).uniform(1.2, 3.5)
        if self.time_scale:
            time.sleep(min(seconds * self.time_scale, 0.4))
        return seconds

    def memory_snapshot(self, model: str):
        sizes = {"demo-local-3b": 2.4, "demo-local-8b": 5.6}
        if model not in sizes:
            return None
        return {"memory_gb": sizes[model], "vram_gb": sizes[model], "gpu_share": 1.0}
