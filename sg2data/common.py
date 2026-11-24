"""Shared paths, text checks and IO helpers for the v2 dataset pipeline."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SOURCES = DATA / "sources"
LARGE = SOURCES / "large"          # raw dumps over 20 MB; git-ignored
CANDIDATES = DATA / "candidates"
LLM_UNITS = DATA / "llm_units"
LLM_OUTPUTS = DATA / "llm_outputs"
JUDGED = DATA / "judged"

SEED = 42
CATEGORIES = ["idioms", "facts", "common_sense", "creative", "ood"]
BLANK = "____"
ZIPF_MIN, ZIPF_MAX = 3.5, 6.0

# Exactly what Qwen sees (DESIGN_v2.md, "Prompts given to Qwen"). The judge is shown
# the same baseline message so it answers the question Qwen answers.
BASELINE_TEMPLATE = "Answer with exactly one English word.\nQuestion: {question}\nAnswer:"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=False) + "\n")
            n += 1
    return n


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_id(category: str, question: str, target: str) -> str:
    """Stable id, same scheme as v1: {category}_{target}_{sha256(question||target||category)[:8]}."""
    digest = hashlib.sha256(f"{question}||{target}||{category}".encode("utf-8")).hexdigest()
    return f"{category}_{target.lower()}_{digest[:8]}"


# ---------------------------------------------------------------- text checks

_WORD_RE = re.compile(r"[A-Za-z]+")


def word_in_text(word: str, text: str) -> bool:
    """Whole-word, case-insensitive match (boundaries are non-alphanumeric chars)."""
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", text, re.IGNORECASE) is not None


def inflections(word: str) -> set[str]:
    """Regular English inflections and the most common derivations of a word.

    Used to catch leaks such as target "run" with "running" in the question. Irregular
    forms are left to the LLM judge.
    """
    w = word.lower()
    forms = {w, w + "s", w + "es", w + "ed", w + "d", w + "ing", w + "er", w + "ers",
             w + "est", w + "ly", w + "ness", w + "ful"}
    if w.endswith("e"):
        forms |= {w[:-1] + "ing", w[:-1] + "er", w[:-1] + "y"}
    if w.endswith("y") and len(w) > 2:
        forms |= {w[:-1] + "ies", w[:-1] + "ied", w[:-1] + "ier", w[:-1] + "iest", w[:-1] + "ily"}
    if len(w) >= 3 and w[-1] not in "aeiouwxy" and w[-2] in "aeiou" and w[-3] not in "aeiou":
        forms |= {w + w[-1] + "ed", w + w[-1] + "ing", w + w[-1] + "er"}
    if w.endswith("s"):
        forms.add(w[:-1])
    if w.endswith("es"):
        forms.add(w[:-2])
    if w.endswith("ies"):
        forms.add(w[:-3] + "y")
    return {f for f in forms if len(f) >= 2}


def leak_reason(target: str, question: str) -> str | None:
    """Return why the question leaks the target, or None.

    Checks: the target as a whole word; any regular inflection of the target; any
    question word whose inflections include the target (e.g. "lights" for "light");
    and, for targets of 4+ letters, any question word that starts with the target
    (compounds such as "lighthouse" for "light").
    """
    t = target.lower()
    if word_in_text(t, question):
        return "target_word"
    forms = inflections(t)
    for tok in _WORD_RE.findall(question.lower()):
        if tok == t:
            return "target_word"
        if tok in forms:
            return f"inflection:{tok}"
        if t in inflections(tok):
            return f"inflection:{tok}"
        if len(t) >= 4 and tok.startswith(t) and len(tok) > len(t):
            return f"compound:{tok}"
    return None


def is_ascii(text: str) -> bool:
    return all(ord(c) < 128 for c in text)


def code_filter(question: str, target: str, max_chars: int = 400) -> str | None:
    """DESIGN_v2 code filters. Return a rejection reason or None if the candidate passes."""
    if not re.fullmatch(r"[A-Za-z]+", target) or len(target) < 2:
        return "target_not_alpha"
    if not is_ascii(question):
        return "non_ascii"
    if question.count(BLANK) != 1:
        return "blank_count"
    if not (len(question) >= 20 and len(question) <= max_chars):
        return "length"
    q = question.lower()
    ends_with_blank = re.search(r"____[.!?\"']*$", question.strip()) is not None
    asks_one_word = "one word" in q or "one-word" in q
    if not (ends_with_blank or asks_one_word):
        return "no_blank_or_one_word"
    reason = leak_reason(target, question)
    if reason:
        return "leak:" + reason
    return None
