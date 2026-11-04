"""Transparent, pluggable accuracy scoring.

Each scorer is a plain function ``(response, item, ctx) -> ScoreResult`` that
explains its verdict in ``detail``, so the dashboard can show *why* an answer
was marked right or wrong. Add a new method with ``@register("name")``.

Built-in methods
----------------
exact        normalized string equality (case, punctuation, articles ignored)
contains     an expected answer appears as a whole word/phrase
numeric      compares the final number in the response (handles $ and 1,234.50)
choice       first option word mentioned (e.g. yes/no, A/B/C/D) must equal expected
regex        a regular expression must match
json         response parses as JSON and contains the expected key/value pairs
instruction  programmatic format checks (IFEval-style): lowercase, word counts, ...
keywords     required / any-of keywords for open-ended answers
fuzzy        token-overlap F1 against the reference answer
judge        an LLM grades the answer against the reference (falls back if no judge)
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from slmbench.schemas import PromptItem, ScoreResult

ScorerFn = Callable[[str, PromptItem, "ScoringContext"], ScoreResult]
_REGISTRY: dict[str, ScorerFn] = {}

SCORER_DESCRIPTIONS = {
    "exact": "Normalized string equality (case, punctuation and articles ignored).",
    "contains": "An expected answer appears as a whole word or phrase anywhere in the response.",
    "numeric": "The last number in the response equals the expected value (handles $ and 1,234.50).",
    "choice": "The first option mentioned (e.g. yes/no, A–D) must be the expected one.",
    "regex": "A regular expression must match the response.",
    "json": "Response parses as a JSON object; partial credit per expected key/value.",
    "instruction": "Programmatic format checks (IFEval-style): lowercase, word/sentence/item counts, …; partial credit per check.",
    "keywords": "Open-ended answers must mention one term from each keyword group; partial credit per group.",
    "fuzzy": "Token-overlap F1 against the reference answer (SQuAD-style), pass at ≥ 0.5.",
    "judge": "An LLM grades the answer against a reference. Falls back to the item's fallback method if no judge is set.",
}


@dataclass
class ScoringContext:
    # Takes a fully-formed grading prompt, returns the judge model's raw text.
    judge: Callable[[str], str] | None = None


def register(name: str) -> Callable[[ScorerFn], ScorerFn]:
    def deco(fn: ScorerFn) -> ScorerFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def available_scorers() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------- helpers

_THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def clean_response(text: str) -> str:
    """Strip chain-of-thought blocks some local models (qwen3, deepseek-r1) emit inline."""
    return _THINK_BLOCK.sub("", text or "").strip()


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation/markdown and articles."""
    text = str(text).lower()
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    nums = []
    for match in _NUMBER.findall(text):
        value = _to_float(match.rstrip(","))
        if value is not None:
            nums.append(value)
    return nums


def _contains_phrase(haystack_norm: str, needle: Any, *, prefix: bool = False) -> bool:
    """Whole-word/phrase match. ``prefix=True`` lets the last word take a suffix
    ("scatter" matches "scattering"), a cheap stand-in for stemming."""
    needle_norm = normalize(needle)
    if not needle_norm:
        return False
    tail = r"\w*" if prefix else r"(?!\w)"
    return re.search(rf"(?<!\w){re.escape(needle_norm)}{tail}", haystack_norm) is not None


def _token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def _strip_code_fence(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text.strip()


def _result(score: float, method: str, detail: str, threshold: float = 1.0) -> ScoreResult:
    return ScoreResult(score=round(score, 4), correct=score >= threshold, method=method, detail=detail)


# --------------------------------------------------------------------------- method selection


def resolve_method(item: PromptItem) -> str:
    """Pick a scorer when the dataset didn't name one."""
    if item.scorer:
        return item.scorer
    expected = item.expected
    if expected is None:
        return "none"
    if isinstance(expected, dict):
        return "json"
    values = _as_list(expected)
    if values and all(_to_float(v) is not None for v in values):
        return "numeric"
    if all(len(str(v).split()) <= 5 for v in values):
        return "contains"
    return "judge"


def score(response: str, item: PromptItem, ctx: ScoringContext | None = None) -> ScoreResult:
    ctx = ctx or ScoringContext()
    method = resolve_method(item)
    if method == "none":
        return ScoreResult.unscored()
    prefix = ""
    if method == "judge" and ctx.judge is None:
        method = item.scorer_args.get("fallback", "fuzzy")
        prefix = "no judge model → "
    fn = _REGISTRY.get(method)
    if fn is None:
        return ScoreResult(None, None, method, f"unknown scorer '{method}'")
    try:
        result = fn(clean_response(response), item, ctx)
    except Exception as exc:  # a buggy custom scorer shouldn't kill the run
        return ScoreResult(None, None, method, f"scorer error: {exc}")
    if prefix:
        result.detail = prefix + result.detail
    return result


def needs_judge(item: PromptItem, ctx: ScoringContext | None) -> bool:
    return resolve_method(item) == "judge" and ctx is not None and ctx.judge is not None


# --------------------------------------------------------------------------- scorers


@register("exact")
def score_exact(response: str, item: PromptItem, ctx: ScoringContext) -> ScoreResult:
    got = normalize(response)
    for expected in _as_list(item.expected):
        if got == normalize(expected):
            return _result(1.0, "exact", f"matched '{expected}'")
    return _result(0.0, "exact", f"expected one of {_as_list(item.expected)}")


@register("contains")
def score_contains(response: str, item: PromptItem, ctx: ScoringContext) -> ScoreResult:
    got = normalize(response)
    for expected in _as_list(item.expected):
        if _contains_phrase(got, expected):
            return _result(1.0, "contains", f"found '{expected}'")
    return _result(0.0, "contains", f"none of {_as_list(item.expected)} found")


@register("numeric")
def score_numeric(response: str, item: PromptItem, ctx: ScoringContext) -> ScoreResult:
    args = item.scorer_args
    targets = [t for t in (_to_float(v) for v in _as_list(item.expected)) if t is not None]
    nums = extract_numbers(response)
    if not nums:
        return _result(0.0, "numeric", "no number in response")
    tol = float(args.get("tolerance", 1e-6))
    match_mode = args.get("match", "last")  # last | first | any
    candidates = {"last": nums[-1:], "first": nums[:1]}.get(match_mode, nums)
    for got in candidates:
        for target in targets:
            if abs(got - target) <= tol:
                return _result(1.0, "numeric", f"{match_mode} number {got:g} == {target:g}")
    shown = ", ".join(f"{n:g}" for n in candidates)
    return _result(0.0, "numeric", f"{match_mode} number {shown} ≠ {' / '.join(f'{t:g}' for t in targets)}")


@register("choice")
def score_choice(response: str, item: PromptItem, ctx: ScoringContext) -> ScoreResult:
    options = item.scorer_args.get("options") or _as_list(item.expected)
    got = normalize(response)
    first: tuple[int, str] | None = None
    for opt in options:
        m = re.search(rf"(?<!\w){re.escape(normalize(opt))}(?!\w)", got)
        if m and (first is None or m.start() < first[0]):
            first = (m.start(), opt)
    if first is None:
        return _result(0.0, "choice", f"no option from {options} found")
    expected = {normalize(e) for e in _as_list(item.expected)}
    ok = normalize(first[1]) in expected
    return _result(1.0 if ok else 0.0, "choice", f"answered '{first[1]}', expected {_as_list(item.expected)}")


@register("regex")
def score_regex(response: str, item: PromptItem, ctx: ScoringContext) -> ScoreResult:
    pattern = item.scorer_args.get("pattern") or str(item.expected)
    flags = 0 if item.scorer_args.get("case_sensitive") else re.IGNORECASE
    ok = re.search(pattern, response, flags | re.DOTALL) is not None
    return _result(1.0 if ok else 0.0, "regex", f"/{pattern}/ {'matched' if ok else 'did not match'}")


@register("json")
def score_json(response: str, item: PromptItem, ctx: ScoringContext) -> ScoreResult:
    """Partial credit per expected key. Code fences are tolerated (most chat models add them)."""
    expected = item.expected or {}
    body = _strip_code_fence(response)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", body, re.DOTALL)
        try:
            parsed = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            parsed = None
    if not isinstance(parsed, dict):
        return _result(0.0, "json", "response is not a JSON object")
    hits, misses = 0, []
    for key, value in expected.items():
        if str(parsed.get(key, "")).strip().lower() == str(value).strip().lower():
            hits += 1
        else:
            misses.append(key)
    extra = sorted(set(parsed) - set(expected)) if item.scorer_args.get("exact_keys") else []
    total = len(expected) + (1 if item.scorer_args.get("exact_keys") else 0)
    passed = hits + (1 if item.scorer_args.get("exact_keys") and not extra else 0)
    detail = "all keys match" if not misses and not extra else f"wrong/missing: {misses}" + (f", extra: {extra}" if extra else "")
    return _result(passed / max(total, 1), "json", detail)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


_CHECKS: dict[str, Callable[[str, dict[str, Any]], bool]] = {
    "lowercase": lambda t, c: any(ch.isalpha() for ch in t) and t == t.lower(),
    "no_commas": lambda t, c: "," not in t,
    "word_count": lambda t, c: c.get("min", 0) <= len(t.split()) <= c.get("max", 10**9),
    "sentence_count": lambda t, c: c.get("min", 0) <= len(_sentences(t)) <= c.get("max", 10**9),
    "item_count": lambda t, c: len([x for x in t.split(c.get("sep", ",")) if x.strip()]) == c["n"],
    "contains_all": lambda t, c: all(_contains_phrase(normalize(t), v) for v in c["values"]),
    "contains_none": lambda t, c: not any(_contains_phrase(normalize(t), v) for v in c["values"]),
    "regex": lambda t, c: re.search(c["pattern"], t, re.IGNORECASE | re.DOTALL) is not None,
    "starts_with": lambda t, c: t.lower().startswith(str(c["value"]).lower()),
    "ends_with": lambda t, c: t.lower().rstrip().endswith(str(c["value"]).lower()),
}


@register("instruction")
def score_instruction(response: str, item: PromptItem, ctx: ScoringContext) -> ScoreResult:
    checks = item.scorer_args.get("checks", [])
    if not checks:
        return ScoreResult(None, None, "instruction", "no checks defined")
    failed = []
    for check in checks:
        fn = _CHECKS.get(check["type"])
        if fn is None or not fn(response, check):
            failed.append(check["type"])
    passed = len(checks) - len(failed)
    detail = f"{passed}/{len(checks)} checks passed" + (f"; failed: {', '.join(failed)}" if failed else "")
    return _result(passed / len(checks), "instruction", detail)


@register("keywords")
def score_keywords(response: str, item: PromptItem, ctx: ScoringContext) -> ScoreResult:
    """Each keyword *group* is satisfied by any one of its synonyms.

    ``groups=[["scatter", "rayleigh"], ["wavelength"]]`` → partial credit per group.
    ``all`` (every term required) and ``any`` (one group) are shorthands.
    Keywords match word prefixes, so "scatter" also covers "scattering".
    """
    args = item.scorer_args
    groups = [list(g) for g in args.get("groups", [])] + [[k] for k in args.get("all", [])]
    if args.get("any"):
        groups.append(list(args["any"]))
    if not groups:
        return ScoreResult(None, None, "keywords", "no keywords defined")
    norm = normalize(response)
    missing = [g for g in groups if not any(_contains_phrase(norm, k, prefix=True) for k in g)]
    passed = len(groups) - len(missing)
    detail = f"{passed}/{len(groups)} keyword groups hit"
    if missing:
        detail += "; missing any of " + " | ".join("/".join(g[:3]) for g in missing)
    return _result(passed / len(groups), "keywords", detail)


@register("fuzzy")
def score_fuzzy(response: str, item: PromptItem, ctx: ScoringContext) -> ScoreResult:
    references = [r for r in [*_as_list(item.expected), item.reference] if r]
    if not references:
        return ScoreResult(None, None, "fuzzy", "no reference answer")
    best = max(_token_f1(response, str(r)) for r in references)
    threshold = float(item.scorer_args.get("threshold", 0.5))
    return _result(best, "fuzzy", f"token F1 {best:.2f} (pass ≥ {threshold:.2f})", threshold=threshold)


JUDGE_TEMPLATE = """You are grading an AI assistant's answer for correctness.

Question:
{question}

Reference answer:
{reference}

Assistant's answer:
{answer}

Is the assistant's answer correct and consistent with the reference? Minor wording
differences are fine; missing or wrong facts are not.
Reply with exactly one word on the first line, CORRECT or INCORRECT, then a one-sentence reason."""


@register("judge")
def score_judge(response: str, item: PromptItem, ctx: ScoringContext) -> ScoreResult:
    if ctx.judge is None:  # only reachable when called directly
        return ScoreResult(None, None, "judge", "no judge configured")
    reference = item.reference or ", ".join(map(str, _as_list(item.expected)))
    verdict = clean_response(ctx.judge(JUDGE_TEMPLATE.format(question=item.prompt, reference=reference, answer=response)))
    first_line, _, reason = verdict.partition("\n")
    word = first_line.strip().strip("*").upper()
    if word.startswith("INCORRECT"):
        ok = False
    elif word.startswith("CORRECT"):
        ok = True
    else:
        return ScoreResult(None, None, "judge", f"unparseable verdict: {verdict[:80]!r}")
    return _result(1.0 if ok else 0.0, "judge", (reason.strip() or first_line.strip())[:200])
