"""Pressure probe: exact baseline P0 for every accepted candidate (measurement 1, baseline
condition only), so the final 2,500 prompts can be balanced over P0 bins before the full
run. Uses the same prompt encoding and scoring code as run_experiment.

    python -m sg2.probe_p0 --candidates data/accepted_candidates.jsonl --out runs/probe_p0.jsonl
"""
import argparse, json, os, time

from . import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="jsonl with prompt_id, category, question, target")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--score-batch", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=1024)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.candidates) if l.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["prompt_id"] for l in open(a.out) if l.strip()}
    rows = [r for r in rows if r["prompt_id"] not in done]
    t0 = time.time()
    lm = core.load_lm(a.model, a.device, a.dtype)
    print(f"loaded {a.model} on {lm.device} in {time.time() - t0:.0f}s; {len(rows)} to probe", flush=True)
    for s in range(0, len(rows), a.chunk):
        chunk = rows[s:s + a.chunk]
        ids = [core.encode_prompt(lm.tok, r["question"], r["target"], "base").ids for r in chunk]
        sc = core.score_variants(lm, ids, [r["target"] for r in chunk], a.score_batch)
        with open(a.out, "a") as fh:
            for r, (p0, per, vs, vids) in zip(chunk, sc):
                fh.write(json.dumps({"prompt_id": r["prompt_id"], "category": r["category"],
                                     "target": r["target"], "p0": p0,
                                     "p0_bin": core.p0_bin(p0)}) + "\n")
        print(f"{s + len(chunk)}/{len(rows)} probed, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
