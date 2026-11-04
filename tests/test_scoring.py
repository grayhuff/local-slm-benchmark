import pytest

from slmbench import scoring
from slmbench.schemas import PromptItem
from slmbench.suites import load_builtin_suite


def item(**kw) -> PromptItem:
    return PromptItem(id="t", prompt="q", **kw)


@pytest.mark.parametrize(
    ("response", "expected", "correct"),
    [
        ("The ball costs 5 cents.", [5, 0.05], True),
        ("$0.05", [5, 0.05], True),
        ("10", [5, 0.05], False),
        ("Total due: $1,284.50", 1284.5, True),
        ("I don't know", 5, False),
    ],
)
def test_numeric(response, expected, correct):
    assert scoring.score(response, item(expected=expected, scorer="numeric")).correct is correct


def test_numeric_uses_last_number_by_default():
    it = item(expected=5, scorer="numeric")
    assert scoring.score("Not 10 — the answer is 5", it).correct
    assert not scoring.score("5 would be wrong; it's 10", it).correct


def test_contains_respects_word_boundaries():
    it = item(expected="Au", scorer="contains")
    assert scoring.score("The symbol is **Au**.", it).correct
    assert not scoring.score("Australia", it).correct


def test_exact_normalizes_case_punctuation_and_articles():
    assert scoring.score("The Canberra.", item(expected="canberra", scorer="exact")).correct


def test_choice_takes_first_option_mentioned():
    it = item(expected="no", scorer="choice", scorer_args={"options": ["yes", "no"]})
    assert scoring.score("No. Carol is the youngest, so yes, Alice is older.", it).correct
    assert not scoring.score("Yes, she is.", it).correct


def test_json_tolerates_code_fences_and_gives_partial_credit():
    it = item(expected={"name": "Maria Chen", "email": "maria.chen@example.com"}, scorer="json", scorer_args={"exact_keys": True})
    good = scoring.score('```json\n{"name": "Maria Chen", "email": "maria.chen@example.com"}\n```', it)
    assert good.correct and good.score == 1.0
    partial = scoring.score('{"name": "Dev", "email": "maria.chen@example.com"}', it)
    assert not partial.correct and 0 < partial.score < 1
    assert scoring.score("not json", it).score == 0.0


def test_instruction_checks_partial_credit():
    it = item(
        scorer="instruction",
        scorer_args={"checks": [{"type": "lowercase"}, {"type": "no_commas"}, {"type": "sentence_count", "min": 1, "max": 1}]},
    )
    assert scoring.score("the sea is calm tonight.", it).correct
    result = scoring.score("The sea is calm, tonight.", it)
    assert result.score == pytest.approx(1 / 3, abs=1e-3)
    assert "lowercase" in result.detail and "no_commas" in result.detail


def test_keywords_groups():
    it = item(scorer="keywords", scorer_args={"groups": [["scatter", "rayleigh"], ["wavelength", "shorter"]]})
    assert scoring.score("Rayleigh scattering affects shorter wavelengths more.", it).correct
    assert scoring.score("Light scatters in the air.", it).score == 0.5


def test_think_blocks_are_stripped_before_scoring():
    it = item(expected=5, scorer="numeric")
    assert scoring.score("<think>maybe 10? no wait</think>\n5", it).correct


def test_judge_falls_back_without_a_judge_model():
    it = item(expected=None, reference="x", scorer="judge", scorer_args={"fallback": "keywords", "any": ["scatter"]})
    result = scoring.score("Molecules scatter blue light.", it)
    assert result.correct and result.method == "keywords" and result.detail.startswith("no judge model")


def test_judge_parses_verdict():
    it = item(reference="Blue light scatters more.", scorer="judge")
    ctx = scoring.ScoringContext(judge=lambda prompt: "CORRECT\nMatches the reference.")
    assert scoring.score("Because of scattering.", it, ctx).correct
    ctx_bad = scoring.ScoringContext(judge=lambda prompt: "INCORRECT\nWrong mechanism.")
    assert scoring.score("Ocean reflection.", it, ctx_bad).correct is False


def test_auto_method_selection():
    assert scoring.resolve_method(item(expected=42)) == "numeric"
    assert scoring.resolve_method(item(expected="Paris")) == "contains"
    assert scoring.resolve_method(item(expected={"a": 1})) == "json"
    assert scoring.resolve_method(item(expected="a long reference answer with many words")) == "judge"
    assert scoring.resolve_method(item()) == "none"
    assert scoring.score("anything", item()).score is None


def test_custom_scorer_registration():
    @scoring.register("always_right")
    def _always(response, it, ctx):
        return scoring.ScoreResult(1.0, True, "always_right")

    assert scoring.score("whatever", item(scorer="always_right")).correct


def test_builtin_suite_references_score_perfectly_and_distractors_do_not():
    """Guards the built-in suite: every reference answer must pass its own scorer."""
    for it in load_builtin_suite():
        assert it.reference, it.id
        ref = scoring.score(it.reference, it)
        assert ref.score == 1.0, (it.id, ref)
        if it.distractor:
            assert scoring.score(it.distractor, it).score < 1.0, it.id
