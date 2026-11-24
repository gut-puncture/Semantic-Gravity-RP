"""Parse and validate LLM unit outputs; flag units to rerun; write candidates / judged files.

Run:
  .venv/bin/python -m sg2data.llm_collect gen   [--model NAME]
  .venv/bin/python -m sg2data.llm_collect judge [--model NAME] [--categories ...]

Outputs are read from data/llm_outputs/{gen,judge}/<unit_id>.json (the model's reply: a
JSON object, optionally wrapped in a ``` fence). A unit whose output is missing, is not
valid JSON, fails the schema, or breaks a semantic rule (wrong count, wrong ids, out-of-range
score) is listed in data/llm_outputs/{kind}/rerun.txt; rerun those units and collect again.
Collecting is idempotent: it always rebuilds from all outputs present.

gen   -> data/candidates/{creative,ood}.jsonl
judge -> data/judged/{category}.jsonl  (every judged candidate, with `accepted`)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from jsonschema import Draft202012Validator

from .common import (BLANK, CANDIDATES, JUDGED, LLM_OUTPUTS, LLM_UNITS, code_filter, prompt_id,
                     read_jsonl, write_jsonl)
from .prompts import CREATIVE_TEMPLATE, GEN_SCHEMA, JUDGE_SCHEMA, K_PER_TARGET, OOD_TEMPLATE

# DESIGN_v2 acceptance rules (v1 thresholds)
MAX_AMBIGUITY = 3
MIN_NATURALNESS = 7


def parse_output(text: str):
    text = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text)


def load_output(kind: str, unit_id: str):
    """Return (payload, error). error is None when the file parsed."""
    path = LLM_OUTPUTS / kind / f"{unit_id}.json"
    if not path.exists():
        return None, "missing"
    try:
        return parse_output(path.read_text(encoding="utf-8")), None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None, f"invalid_json: {e}"


def schema_errors(payload, schema) -> list[str]:
    return [e.message for e in Draft202012Validator(schema).iter_errors(payload)][:3]


def write_rerun(kind: str, bad: dict[str, str]) -> None:
    out = LLM_OUTPUTS / kind
    out.mkdir(parents=True, exist_ok=True)
    (out / "rerun.txt").write_text("".join(f"{u}\t{why}\n" for u, why in sorted(bad.items())))


# ================================================================== generation

def check_gen(unit: dict, payload) -> str | None:
    errs = schema_errors(payload, GEN_SCHEMA)
    if errs:
        return "schema: " + "; ".join(errs)
    if payload["target_word"].strip().lower() != unit["target_word"]:
        return "wrong_target_word"
    if not payload["target_usable"]:
        return None if not payload["items"] else "unusable_but_items"
    if len(payload["items"]) != unit["k"]:
        return f"expected_{unit['k']}_items_got_{len(payload['items'])}"
    for it in payload["items"]:
        if it["text"].count(BLANK) != 1:
            return "item_without_exactly_one_blank"
    return None


def collect_gen(model: str | None) -> dict:
    bad: dict[str, str] = {}
    report: dict = {}
    for category, template in [("creative", CREATIVE_TEMPLATE), ("ood", OOD_TEMPLATE)]:
        units = sorted((LLM_UNITS / "gen").glob(f"unit_{category}_*.json"))
        rows, rejects, unusable = [], Counter(), []
        seen = set()
        for path in units:
            unit = json.loads(path.read_text())
            payload, err = load_output("gen", unit["unit_id"])
            err = err or check_gen(unit, payload)
            if err:
                bad[unit["unit_id"]] = err
                continue
            if not payload["target_usable"]:
                unusable.append({"word": unit["target_word"], "reason": payload["unusable_reason"]})
                continue
            for idx, it in enumerate(payload["items"]):
                text = " ".join(it["text"].split())
                if category == "creative" and not re.search(r"____[.!?]$", text):
                    rejects["creative_blank_not_final"] += 1
                    continue
                question = template.format(text=text)
                target = unit["target_word"]
                reason = code_filter(question, target)
                if reason:
                    rejects[reason.split(":")[0] if reason.startswith("leak") else reason] += 1
                    continue
                if question.lower() in seen:
                    rejects["duplicate"] += 1
                    continue
                seen.add(question.lower())
                rows.append({
                    "prompt_id": prompt_id(category, question, target), "category": category,
                    "question": question, "target": target,
                    "template_id": "D_creative" if category == "creative" else "E_ood",
                    "source_trace": {"source": "llm_generation", "unit_id": unit["unit_id"],
                                     "item_index": idx, "setting": it["setting"],
                                     "assigned_setting": unit["assigned_settings"][idx]
                                     if idx < len(unit["assigned_settings"]) else None,
                                     "model": model}})
        write_jsonl(CANDIDATES / f"{category}.jsonl", rows)
        (CANDIDATES / f"unusable_targets_{category}.json").write_text(json.dumps(unusable, indent=1))
        report[category] = {"units": len(units),
                            "units_ok": len(units) - sum(u.startswith(f"unit_{category}_") for u in bad),
                            "targets_unusable": len(unusable), "candidates": len(rows),
                            "targets_with_candidates": len({r["target"] for r in rows}),
                            "item_rejections": dict(rejects)}
    write_rerun("gen", bad)
    report["units_to_rerun"] = len(bad)
    (LLM_OUTPUTS / "gen" / "collect_report.json").write_text(json.dumps(report, indent=2))
    return report


# ================================================================== judging

def check_judge(unit: dict, payload) -> str | None:
    errs = schema_errors(payload, JUDGE_SCHEMA)
    if errs:
        return "schema: " + "; ".join(errs)
    ids = [it["id"] for it in payload["items"]]
    if sorted(ids) != sorted(unit["ids"]):
        return "ids_mismatch"
    for it in payload["items"]:
        if not (0 <= it["ambiguity"] <= 10 and 0 <= it["naturalness"] <= 10):
            return "score_out_of_range"
        if len(it["top3"]) != 3:
            return "top3_not_three"
        if not re.fullmatch(r"[A-Za-z]+", it["best_one_word_answer"].strip()):
            return "best_not_one_word"
    return None


def norm(w: str) -> str:
    return w.strip().strip(".,!?\"'").lower()


def collect_judge(categories: list[str], model: str | None) -> dict:
    bad: dict[str, str] = {}
    report: dict = {}
    for category in categories:
        cpath = CANDIDATES / f"{category}.jsonl"
        if not cpath.exists():
            continue
        cands = {r["prompt_id"]: r for r in read_jsonl(cpath)}
        units = sorted((LLM_UNITS / "judge").glob(f"unit_{category}_*.json"))
        rows = []
        for path in units:
            unit = json.loads(path.read_text())
            payload, err = load_output("judge", unit["unit_id"])
            err = err or check_judge(unit, payload)
            if err:
                bad[unit["unit_id"]] = err
                continue
            for it in payload["items"]:
                c = cands.get(unit["ids"][it["id"]])
                if c is None:   # candidates were rebuilt after the units: stale unit
                    bad[unit["unit_id"]] = "stale_unit_prompt_id_not_in_candidates"
                    break
                best = it["best_one_word_answer"].strip()
                top3 = [t.strip() for t in it["top3"]]
                is_x_best = norm(best) == norm(c["target"])
                accepted = (is_x_best and it["ambiguity"] <= MAX_AMBIGUITY
                            and not it["leaks_answer"] and it["naturalness"] >= MIN_NATURALNESS
                            and not it["offensive"])
                rows.append({**c, "judge": {
                    "best_one_word_answer": best, "top3": top3, "is_X_best": is_x_best,
                    "target_rank_in_top3": next((i for i, t in enumerate(top3) if norm(t) == norm(c["target"])), -1),
                    "ambiguity": it["ambiguity"], "leaks_answer": it["leaks_answer"],
                    "naturalness": it["naturalness"], "offensive": it["offensive"], "unit_id": unit["unit_id"], "model": model,
                    "blind": True}, "accepted": accepted})
        JUDGED.mkdir(parents=True, exist_ok=True)
        write_jsonl(JUDGED / f"{category}.jsonl", rows)
        acc = [r for r in rows if r["accepted"]]
        report[category] = {
            "candidates": len(cands), "judged": len(rows), "not_judged": len(cands) - len(rows),
            "accepted": len(acc), "accepted_distinct_targets": len({r["target"].lower() for r in acc}),
            "fail_reasons": dict(Counter(
                ("not_X_best" if not r["judge"]["is_X_best"] else
                 "leak" if r["judge"]["leaks_answer"] else
                 "ambiguous" if r["judge"]["ambiguity"] > MAX_AMBIGUITY else
                 "unnatural" if r["judge"]["naturalness"] < MIN_NATURALNESS else "offensive")
                for r in rows if not r["accepted"]))}
    write_rerun("judge", bad)
    report["units_to_rerun"] = len(bad)
    (LLM_OUTPUTS / "judge").mkdir(parents=True, exist_ok=True)
    (LLM_OUTPUTS / "judge" / "collect_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["gen", "judge"])
    ap.add_argument("--categories", default="idioms,facts,common_sense,creative,ood")
    ap.add_argument("--model", default=None, help="model name recorded in source_trace / judge")
    args = ap.parse_args(argv)
    rep = collect_gen(args.model) if args.kind == "gen" else collect_judge(args.categories.split(","), args.model)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
