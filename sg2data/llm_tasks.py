"""Build (but never send) the LLM work units.

Run:
  .venv/bin/python -m sg2data.llm_tasks gen                 # creative/ood generation units
  .venv/bin/python -m sg2data.llm_tasks judge [--categories idioms,facts,common_sense]

Each unit is one self-contained JSON file under data/llm_units/{gen,judge}/:
  unit_id, kind, category, prompt (the full text to send), output_schema (JSON Schema),
  and a mapping used only by llm_collect.py (judge: item id -> prompt_id; targets are
  NOT in the unit file, so an agent backend cannot read the intended answer).
Run each unit with any backend and save the model's JSON reply to
data/llm_outputs/{gen,judge}/<unit_id>.json (see README).

Judge units need the candidates first, so for creative/ood run `gen`, the generation
units, and `llm_collect gen` before `judge --categories creative,ood`.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter

from .common import CANDIDATES, LLM_UNITS, SEED, read_jsonl
from .prompts import (COMMON_SENSE_GEN, COMMON_SENSE_SETTINGS, CREATIVE_GEN, CREATIVE_SETTINGS, GEN_SCHEMA, JUDGE, JUDGE_CATEGORY_NOTES,
                      JUDGE_SCHEMA, K_PER_TARGET, OOD_FRAMINGS, OOD_GEN)

JUDGE_BATCH = 25


def _write(path, unit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(unit, indent=1, ensure_ascii=True))


GEN_CATEGORIES = {"creative": (CREATIVE_GEN, CREATIVE_SETTINGS), "ood": (OOD_GEN, OOD_FRAMINGS),
                  "common_sense": (COMMON_SENSE_GEN, COMMON_SENSE_SETTINGS)}


def build_gen(categories: list[str]) -> dict:
    """(Re)write the generation units of the given categories only; other categories' units
    and all outputs are left alone."""
    out_dir = LLM_UNITS / "gen"
    counts = {}
    for category in categories:
        template, pool = GEN_CATEGORIES[category]
        for old in out_dir.glob(f"unit_{category}_*.json"):
            old.unlink()
        words = json.loads((CANDIDATES / f"targets_{category}.json").read_text())["words"]
        rng = random.Random(f"{SEED}:gen:{category}")
        for i, w in enumerate(words, 1):
            settings = rng.sample(pool, K_PER_TARGET)
            prompt = template.replace("{X}", w).replace(
                "{SETTINGS}", "\n".join(f"{j}. {s}" for j, s in enumerate(settings, 1)))
            unit_id = f"unit_{category}_{i:04d}"
            _write(out_dir / f"{unit_id}.json", {
                "unit_id": unit_id, "kind": "gen", "category": category, "target_word": w,
                "assigned_settings": settings, "k": K_PER_TARGET,
                "prompt": prompt, "output_schema": GEN_SCHEMA})
        counts[category] = len(words)
    return counts


def _batches(rows: list[dict], rng: random.Random) -> list[list[dict]]:
    """Seeded shuffle, then deal into batches of <= JUDGE_BATCH so that no batch holds two
    candidates with the same target (siblings would hint at each other's answer)."""
    rows = rows[:]
    rng.shuffle(rows)
    max_same = max(Counter(r["target"].lower() for r in rows).values())
    n_batches = max(-(-len(rows) // JUDGE_BATCH), max_same)
    batches: list[list[dict]] = [[] for _ in range(n_batches)]
    targets: list[set] = [set() for _ in range(n_batches)]
    for k, r in enumerate(rows):
        t = r["target"].lower()
        for off in range(n_batches):
            b = (k + off) % n_batches
            if len(batches[b]) < JUDGE_BATCH and t not in targets[b]:
                batches[b].append(r)
                targets[b].add(t)
                break
        else:  # cannot happen: there are at least as many batches as copies of any target
            raise AssertionError("batching failed")
    return [b for b in batches if b]


def _write_judge_units(category: str, rows: list[dict], start: int, rng: random.Random) -> int:
    out_dir = LLM_UNITS / "judge"
    batches = _batches(rows, rng)
    for i, batch in enumerate(batches, start):
        mapping = {f"q{j:02d}": r["prompt_id"] for j, r in enumerate(batch, 1)}
        items_text = "\n".join(json.dumps({"id": f"q{j:02d}", "question": r["question"]}, ensure_ascii=True)
                               for j, r in enumerate(batch, 1))
        prompt = (JUDGE.replace("{CATEGORY_NOTE}", JUDGE_CATEGORY_NOTES.get(category, ""))
                  .replace("{CATEGORY}", category).replace("{ITEMS}", items_text))
        unit_id = f"unit_{category}_{i:04d}"
        _write(out_dir / f"{unit_id}.json", {
            "unit_id": unit_id, "kind": "judge", "category": category,
            "n_items": len(batch), "ids": mapping, "prompt": prompt, "output_schema": JUDGE_SCHEMA})
    return len(batches)


def build_judge(categories: list[str], append: bool = False) -> dict:
    """Without --append: rewrite all judge units of each category. With --append: keep the
    existing units and add units (numbered after the last one) for candidates that no unit
    covers yet."""
    counts = {}
    out_dir = LLM_UNITS / "judge"
    for category in categories:
        path = CANDIDATES / f"{category}.jsonl"
        if not path.exists():
            print(f"judge: {category}: no candidates yet ({path.name}); skipped")
            continue
        rows = read_jsonl(path)
        existing = sorted(out_dir.glob(f"unit_{category}_*.json"))
        if append:
            covered = {pid for p in existing for pid in json.loads(p.read_text())["ids"].values()}
            rows = [r for r in rows if r["prompt_id"] not in covered]
            start = max((int(p.stem.rsplit("_", 1)[1]) for p in existing), default=0) + 1
            rng = random.Random(f"{SEED}:judge:{category}:from{start}")
        else:
            for old in existing:
                old.unlink()
            start, rng = 1, random.Random(f"{SEED}:judge:{category}")
        n = _write_judge_units(category, rows, start, rng) if rows else 0
        counts[category] = {"candidates": len(rows), "units": n, "first_unit": start}
    return counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["gen", "judge"])
    ap.add_argument("--categories", default=None,
                    help="default: creative,ood,common_sense for gen; all five for judge")
    ap.add_argument("--append", action="store_true",
                    help="judge: only add units for candidates no existing unit covers")
    args = ap.parse_args(argv)
    if args.kind == "gen":
        print(json.dumps(build_gen((args.categories or "creative,ood,common_sense").split(","))))
    else:
        cats = (args.categories or "idioms,facts,common_sense,creative,ood").split(",")
        print(json.dumps(build_judge(cats, append=args.append)))


if __name__ == "__main__":
    main()
