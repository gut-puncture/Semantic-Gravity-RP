"""
prompt_builder.py - Prompt Formatting for Semantic Gravity Experiment

This module provides:
- build_prompt(question_text, target_word, condition) function
- Exact baseline and negative instruction templates per spec

Templates match specification.md Section 3.4 and execution-plan.md Section 7.1.
"""

# ============================================================================
# FALLBACK TEMPLATES (used if config import fails)
# ============================================================================

_BASELINE_TEMPLATE = (
    "Answer with exactly one English word.\n"
    "Question: {question}\n"
    "Answer:"
)

_NEGATIVE_TEMPLATE = (
    "Answer with exactly one English word.\n"
    "Do not use the word \"{target}\" anywhere in your answer.\n"
    "Question: {question}\n"
    "Answer:"
)

# ============================================================================
# TEMPLATE LOADING
# ============================================================================

_templates = None


def _get_templates():
    """Load templates from config or use fallbacks."""
    global _templates
    if _templates is not None:
        return _templates

    try:
        from .config import PROMPT_TEMPLATES
        _templates = {
            "baseline": PROMPT_TEMPLATES.get("baseline", _BASELINE_TEMPLATE),
            "negative": PROMPT_TEMPLATES.get("negative_instruction", _NEGATIVE_TEMPLATE),
        }
    except ImportError:
        try:
            from config import PROMPT_TEMPLATES
            _templates = {
                "baseline": PROMPT_TEMPLATES.get("baseline", _BASELINE_TEMPLATE),
                "negative": PROMPT_TEMPLATES.get("negative_instruction", _NEGATIVE_TEMPLATE),
            }
        except ImportError:
            _templates = {
                "baseline": _BASELINE_TEMPLATE,
                "negative": _NEGATIVE_TEMPLATE,
            }

    return _templates


# ============================================================================
# PUBLIC API
# ============================================================================


def build_prompt(question_text, target_word, condition):
    """
    Build a formatted prompt for the given condition.

    Args:
        question_text: The question text (without the instruction prefix)
        target_word: The target word X that is the expected answer
        condition: Either "baseline" or "negative"

    Returns:
        Formatted prompt string ready for model input

    Raises:
        ValueError: If condition is not "baseline" or "negative"
    """
    if condition not in ("baseline", "negative"):
        raise ValueError(
            f"condition must be 'baseline' or 'negative', got: {condition!r}"
        )

    templates = _get_templates()

    if condition == "baseline":
        return templates["baseline"].format(question=question_text)
    else:
        return templates["negative"].format(question=question_text, target=target_word)


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROMPT BUILDER SELF-TEST")
    print("=" * 60)

    question = "What is 2+2?"
    target = "four"

    # Test baseline
    baseline_result = build_prompt(question, target, "baseline")
    baseline_expected = (
        "Answer with exactly one English word.\n"
        "Question: What is 2+2?\n"
        "Answer:"
    )
    assert baseline_result == baseline_expected, (
        f"Baseline mismatch:\nGot:\n{baseline_result!r}\nExpected:\n{baseline_expected!r}"
    )
    print("Baseline prompt: PASS")

    # Test negative
    negative_result = build_prompt(question, target, "negative")
    negative_expected = (
        "Answer with exactly one English word.\n"
        "Do not use the word \"four\" anywhere in your answer.\n"
        "Question: What is 2+2?\n"
        "Answer:"
    )
    assert negative_result == negative_expected, (
        f"Negative mismatch:\nGot:\n{negative_result!r}\nExpected:\n{negative_expected!r}"
    )
    print("Negative prompt: PASS")

    # Test invalid condition
    try:
        build_prompt(question, target, "invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Invalid condition raises ValueError: PASS")

    print("=" * 60)
    print("All self-tests passed!")
    print("=" * 60)
