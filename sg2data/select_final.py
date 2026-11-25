"""Two steps around the GPU pressure probe.

1. `accepted`: write data/accepted_candidates.jsonl, every judge-accepted candidate of the
   five categories. For the generated categories (creative, ood, common_sense) keep one
   accepted candidate per target word, chosen at random (seed 42), never by P0.
2. `select`: read the probe's P0 per candidate and pick 500 prompts per category,
   balanced over the five P0 bins: 100 per bin where possible; a bin's shortfall is filled
   from the bins nearest to it (all of distance 1 before distance 2; lower bin first on a
   tie), at random within a bin (seed 42). Writes
   data/prompts_v2.jsonl and data/selection_report.json.

    python -m sg2data.select_final accepted
    python -m sg2data.select_final select --probe runs/probe_p0.jsonl
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CATS = ["idioms", "facts", "common_sense", "creative", "ood"]
GENERATED = {"creative", "ood", "common_sense"}
PER_CAT, N_BINS, SEED = 500, 5, 42


def accepted():
    rng = random.Random(SEED)
    out, report = [], {}
    for c in CATS:
        rows = [json.loads(l) for l in open(DATA / "judged" / f"{c}.jsonl")]
        acc = [r for r in rows if r.get("accepted")]
        if c in GENERATED:
            by_t = defaultdict(list)
            for r in acc:
                by_t[r["target"].lower()].append(r)
            acc = [rng.choice(sorted(v, key=lambda r: r["prompt_id"])) for _, v in sorted(by_t.items())]
        report[c] = {"judged": len(rows), "accepted_kept": len(acc)}
        out += [{k: r[k] for k in ("prompt_id", "category", "question", "target", "template_id", "source_trace")}
                for r in acc]
    with open(DATA / "accepted_candidates.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=1))


def select(probe_path, candidates=None, out_dir=None, per_cat=PER_CAT):
    """Per category: quota per_cat/5 per P0 bin. A bin with too few items is short by the
    difference, and its shortfall is filled from bins with spare items, nearest first: every
    short bin takes all it can from bins at distance 1 before any bin at distance 2 is used,
    and so on. Ties are broken deterministically: short bins are served in ascending order,
    and at equal distance the lower bin gives before the higher one. (If per_cat is not a
    multiple of 5, the lowest bins get one extra; production uses 500.) Stops with an error if
    any candidate has no probe P0 or any category cannot reach per_cat."""
    rng = random.Random(SEED)
    p0 = {}
    for l in open(probe_path):
        if l.strip():
            d = json.loads(l)
            p0[d["prompt_id"]] = d["p0"]
    cands = [json.loads(l) for l in open(candidates or DATA / "accepted_candidates.jsonl") if l.strip()]
    missing = [r["prompt_id"] for r in cands if r["prompt_id"] not in p0]
    if missing:
        raise SystemExit(f"STOPPED: {len(missing)} of {len(cands)} candidates have no P0 in "
                         f"{probe_path} (first: {missing[:5]}). Finish the probe first.")
    out_dir = Path(out_dir) if out_dir else DATA
    final, report = [], {}
    for c in CATS:
        bins = defaultdict(list)
        for r in cands:
            if r["category"] == c:
                r = dict(r, p0=p0[r["prompt_id"]])
                bins[min(int(r["p0"] * N_BINS), N_BINS - 1)].append(r)
        for b in bins.values():
            b.sort(key=lambda r: r["prompt_id"]); rng.shuffle(b)
        avail = {b: len(bins[b]) for b in range(N_BINS)}
        quota = {b: per_cat // N_BINS + (1 if b < per_cat % N_BINS else 0) for b in range(N_BINS)}
        take = {b: min(quota[b], avail[b]) for b in range(N_BINS)}
        need = {b: quota[b] - take[b] for b in range(N_BINS)}
        for dist in range(1, N_BINS):
            for s in range(N_BINS):
                for b in (s - dist, s + dist):
                    if need[s] > 0 and 0 <= b < N_BINS and avail[b] > take[b]:
                        k = min(need[s], avail[b] - take[b])
                        take[b] += k; need[s] -= k
        chosen = [r for b in range(N_BINS) for r in bins[b][:take[b]]]
        report[c] = {"available_per_bin": avail, "selected_per_bin": take, "selected": len(chosen),
                     "shortfall": per_cat - len(chosen)}
        final += chosen
    report["totals"] = {"selected": len(final), "requested": per_cat * len(CATS),
                        "selected_per_category": {c: report[c]["selected"] for c in CATS}}
    (out_dir / "selection_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    bad = {c: report[c]["selected"] for c in CATS if report[c]["shortfall"]}
    if bad:
        raise SystemExit(f"STOPPED: categories below the quota of {per_cat}: {bad}. "
                         "No prompts file written.")
    with open(out_dir / "prompts_v2.jsonl", "w") as fh:
        for r in final:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["accepted", "select"])
    ap.add_argument("--probe")
    ap.add_argument("--candidates", default=None, help="default data/accepted_candidates.jsonl")
    ap.add_argument("--out-dir", default=None, help="where prompts_v2.jsonl and "
                    "selection_report.json go; default data/")
    ap.add_argument("--per-cat", type=int, default=PER_CAT, help="prompts per category (tests)")
    a = ap.parse_args()
    if a.step == "accepted":
        accepted()
    else:
        select(a.probe, a.candidates, a.out_dir, a.per_cat)
