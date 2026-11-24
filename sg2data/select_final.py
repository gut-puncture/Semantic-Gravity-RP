"""Two steps around the GPU pressure probe.

1. `accepted`: write data/accepted_candidates.jsonl, every judge-accepted candidate of the
   five categories. For the generated categories (creative, ood, common_sense) keep one
   accepted candidate per target word, chosen at random (seed 42), never by P0.
2. `select`: read the probe's P0 per candidate and pick 500 prompts per category,
   balanced over the five P0 bins: 100 per bin where possible; a bin's shortfall is filled
   from the other bins nearest to it first, at random within a bin (seed 42). Writes
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


def select(probe_path):
    rng = random.Random(SEED)
    p0 = {}
    for l in open(probe_path):
        d = json.loads(l)
        p0[d["prompt_id"]] = d["p0"]
    cands = [json.loads(l) for l in open(DATA / "accepted_candidates.jsonl")]
    final, report = [], {}
    for c in CATS:
        bins = defaultdict(list)
        for r in cands:
            if r["category"] == c and r["prompt_id"] in p0:
                r = dict(r, p0=p0[r["prompt_id"]])
                bins[min(int(r["p0"] * N_BINS), N_BINS - 1)].append(r)
        for b in bins.values():
            b.sort(key=lambda r: r["prompt_id"]); rng.shuffle(b)
        avail = {b: len(bins[b]) for b in range(N_BINS)}
        take = {b: min(PER_CAT // N_BINS, avail[b]) for b in range(N_BINS)}
        short = PER_CAT - sum(take.values())
        # fill the shortfall from bins with spare items, nearest to the short bins first
        short_bins = [b for b in range(N_BINS) if take[b] < PER_CAT // N_BINS]
        order = sorted(range(N_BINS), key=lambda b: (min([abs(b - s) for s in short_bins] or [0]), b))
        while short > 0:
            moved = False
            for b in order:
                if short > 0 and take[b] < avail[b]:
                    take[b] += 1; short -= 1; moved = True
            if not moved:
                break
        chosen = [r for b in range(N_BINS) for r in bins[b][:take[b]]]
        report[c] = {"available_per_bin": avail, "selected_per_bin": take, "selected": len(chosen),
                     "shortfall": PER_CAT - len(chosen)}
        final += chosen
    with open(DATA / "prompts_v2.jsonl", "w") as fh:
        for r in final:
            fh.write(json.dumps(r) + "\n")
    (DATA / "selection_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["accepted", "select"])
    ap.add_argument("--probe")
    a = ap.parse_args()
    accepted() if a.step == "accepted" else select(a.probe)
