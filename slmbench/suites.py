"""Prompt sources: the built-in suite, user JSONL uploads, a single custom
prompt, and (optionally) a sample from a Hugging Face dataset.

JSONL schema (one object per line; only ``prompt`` is required)::

    {"id": "q1", "category": "math", "prompt": "What is 17 * 24?",
     "expected": 408, "scorer": "numeric", "scorer_args": {"tolerance": 0.01},
     "reference": "408"}

``question``/``input`` are accepted as aliases for ``prompt`` and
``answer``/``target`` for ``expected``, so many existing eval files load as-is.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from slmbench.config import BUILTIN_SUITE_PATH
from slmbench.schemas import PromptItem
from slmbench.scoring import available_scorers

_PROMPT_KEYS = ("prompt", "question", "input")
_EXPECTED_KEYS = ("expected", "answer", "target")


def _first(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if obj.get(k) not in (None, ""):
            return obj[k]
    return None


def item_from_dict(obj: dict[str, Any], default_id: str) -> PromptItem:
    prompt = _first(obj, _PROMPT_KEYS)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("missing a non-empty 'prompt' field")
    scorer = obj.get("scorer") or None
    if scorer and scorer not in available_scorers() and scorer != "none":
        raise ValueError(f"unknown scorer '{scorer}' (available: {', '.join(available_scorers())})")
    return PromptItem(
        id=str(obj.get("id") or default_id),
        prompt=prompt.strip(),
        category=str(obj.get("category") or "custom"),
        expected=_first(obj, _EXPECTED_KEYS),
        scorer=scorer,
        scorer_args=dict(obj.get("scorer_args") or obj.get("args") or {}),
        reference=obj.get("reference"),
        distractor=obj.get("distractor"),
    )


def parse_jsonl(text: str, source: str = "upload") -> tuple[list[PromptItem], list[str]]:
    """Parse JSONL leniently: bad lines are reported, good lines still load."""
    items: list[PromptItem] = []
    errors: list[str] = []
    seen: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("line is not a JSON object")
            item = item_from_dict(obj, default_id=f"{source}-{lineno}")
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"line {lineno}: {exc}")
            continue
        if item.id in seen:
            item.id = f"{item.id}-{lineno}"
        seen.add(item.id)
        items.append(item)
    return items, errors


def load_jsonl_file(path: Path) -> list[PromptItem]:
    items, errors = parse_jsonl(Path(path).read_text(), source=Path(path).stem)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors[:5]))
    return items


def load_builtin_suite() -> list[PromptItem]:
    return load_jsonl_file(BUILTIN_SUITE_PATH)


def custom_prompt_item(prompt: str, expected: str | None = None, scorer: str | None = None) -> PromptItem:
    return PromptItem(
        id="custom-1",
        prompt=prompt.strip(),
        category="custom",
        expected=expected.strip() if expected and expected.strip() else None,
        scorer=scorer or None,
    )


# Hugging Face samples are optional: `pip install datasets` to enable.
HF_DATASETS: dict[str, dict[str, str]] = {
    "gsm8k": {"path": "openai/gsm8k", "config": "main", "split": "test", "label": "GSM8K (grade-school math)"},
}


def load_hf_sample(name: str = "gsm8k", n: int = 10, seed: int = 0) -> list[PromptItem]:
    try:
        from datasets import load_dataset  # lazy: heavy optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install the optional dependency first: pip install datasets") from exc
    spec = HF_DATASETS[name]
    ds = load_dataset(spec["path"], spec["config"], split=spec["split"])
    indices = random.Random(seed).sample(range(len(ds)), k=min(n, len(ds)))
    items = []
    for i in indices:
        row = ds[i]
        # GSM8K answers end with "#### <number>"
        final = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append(
            PromptItem(
                id=f"{name}-{i}",
                prompt=f"{row['question']}\n\nSolve the problem, then give the final answer as a single number on the last line.",
                category=name,
                expected=final,
                scorer="numeric",
                scorer_args={"match": "last", "tolerance": 1e-6},
                reference=final,
            )
        )
    return items
