"""Semantic Gravity v2 GPU run: measurements 1-7 for every prompt, then patching
(measurement 8) on a subset. Spec: DESIGN_v2.md.

    python -m sg2.run_experiment --prompts prompts.jsonl --out runs/v2 \
        --model Qwen/Qwen2.5-7B-Instruct

Resumable: prompts already written are skipped. Results go to <out>/measure/part-*.parquet
and <out>/patch/part-*.parquet; each part is one chunk written atomically, so a crash
loses at most the chunk in progress.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from sg2 import core

GREEDY_MAX_NEW = 8
SAMPLE_MAX_NEW = 10
TEMPERATURE = 1.0
TOP_P = 0.9


# ----------------------------------------------------------------------------- io helpers

def log(out_dir: str, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(out_dir, "run.log"), "a") as f:
        f.write(line + "\n")


def write_part(df: pd.DataFrame, folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    n = len(glob.glob(os.path.join(folder, "part-*.parquet")))
    path = os.path.join(folder, f"part-{n:05d}.parquet")
    while os.path.exists(path):
        n += 1
        path = os.path.join(folder, f"part-{n:05d}.parquet")
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def read_parts(folder: str, columns=None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "part-*.parquet")))
    if not files:
        return pd.DataFrame(columns=columns or [])
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def load_prompts(path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ids = [r["prompt_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate prompt_id in prompts file"
    for r in rows:
        assert not core.contains_word(r["target"], r["question"]), f"target in question: {r}"
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(__file__)).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def check_invariants(out_dir: str, args, lm: core.LM) -> None:
    """The run is one task only if what it measures is unchanged: model, dtype, prompt
    templates, sampling settings, seed. Batch sizes, chunk size, hardware may change."""
    inv = {
        "model": args.model, "dtype": str(next(lm.model.parameters()).dtype),
        "baseline_template": core.BASELINE_TEMPLATE, "negative_template": core.NEGATIVE_TEMPLATE,
        "n_samples": args.n_samples, "temperature": TEMPERATURE, "top_p": TOP_P,
        "greedy_max_new": GREEDY_MAX_NEW, "sample_max_new": SAMPLE_MAX_NEW, "seed": args.seed,
    }
    path = os.path.join(out_dir, "run_config.json")
    record = {"time": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
              "git_commit": git_commit(), "device": str(lm.device),
              "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    if os.path.exists(path):
        cfg = json.load(open(path))
        diff = {k: (cfg["invariants"].get(k), v) for k, v in inv.items()
                if cfg["invariants"].get(k) != v}
        if diff:
            raise SystemExit(f"Refusing to resume: invariants changed {diff}. Use a new --out.")
        cfg["starts"].append(record)
    else:
        cfg = {"invariants": inv, "starts": [record]}
    json.dump(cfg, open(path, "w"), indent=2)


def chunk_seed(seed: int, first_id: str) -> int:
    return int(hashlib.sha256(f"{seed}:{first_id}".encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------- stage 1

def measure_chunk(lm: core.LM, rows: list[dict], args) -> pd.DataFrame:
    tok = lm.tok
    t0 = time.time()
    enc = {c: [core.encode_prompt(tok, r["question"], r["target"], c) for r in rows]
           for c in core.CONDITIONS}
    targets = [r["target"] for r in rows]

    # 1. Exact P0 / P1 (sdpa attention: no weights needed).
    core.set_attn(lm, "sdpa")
    sc = core.score_variants(lm, [e.ids for e in enc["base"]] + [e.ids for e in enc["neg"]],
                             targets + targets, args.score_batch)
    sc_base, sc_neg = sc[: len(rows)], sc[len(rows):]
    t1 = time.time()

    # Target token t*: first token of the surface variant with the highest P0 + P1 (the
    # model's preferred spelling of X across both conditions; the negative prompt often
    # shifts case, e.g. "rain" -> "Rain"). Target set: distinct first tokens of all
    # variants, t* first; its probability is the next-token analogue of P0/P1.
    tstar, tset = [], []
    for (_, pb, vs, vids), (_, pn, _, _) in zip(sc_base, sc_neg):
        best = int(np.argmax(np.add(pb, pn)))
        t = vids[best][0]
        rest = []
        for v in vids:
            if v[0] != t and v[0] not in rest:
                rest.append(v[0])
        tstar.append(t)
        tset.append([t] + rest)

    # 5-7. Logit lens, attention, DLA in one eager pass per condition.
    core.set_attn(lm, "eager")
    res = {}
    for c in core.CONDITIONS:
        items = list(range(len(rows)))

        def fn(idx, c=c):
            seqs = [enc[c][i].ids for i in idx]
            cap, logits = core.hooked_forward(lm, seqs, want_attn=(c == "neg"))
            ts = [tset[i] for i in idx]
            lt, ls = core.lens_probs(lm, cap["resid"], ts)
            ts_t = torch.tensor([tstar[i] for i in idx], device=lm.device)
            d = core.dla(lm, cap, ts_t)
            probs = torch.softmax(logits, -1)
            out = {"lens_tok": lt, "lens_set": ls,
                   "logit_t": logits.gather(1, ts_t[:, None]).squeeze(1).cpu(),
                   "p_t": probs.gather(1, ts_t[:, None]).squeeze(1).cpu(),
                   "top1": logits.argmax(-1).cpu(), **d}
            if c == "neg":
                out.update(core.attention_masses(cap["attw"], [enc[c][i].spans for i in idx]))
            return [{k: v[j] for k, v in out.items()} for j in range(len(idx))]

        res[c] = core.run_split(fn, items, args.fwd_batch)
    core.set_attn(lm, "sdpa")
    t2 = time.time()

    # 2. Greedy, 3. samples (left padding inside core.generate).
    all_ids = [e.ids for e in enc["base"]] + [e.ids for e in enc["neg"]]
    greedy = core.generate(lm, all_ids, do_sample=False, n=1, max_new_tokens=GREEDY_MAX_NEW,
                           seq_batch=args.gen_batch)
    t3 = time.time()
    seed = chunk_seed(args.seed, rows[0]["prompt_id"])
    torch.manual_seed(seed)
    samples = core.generate(lm, all_ids, do_sample=True, n=args.n_samples,
                            max_new_tokens=SAMPLE_MAX_NEW, temperature=TEMPERATURE, top_p=TOP_P,
                            seq_batch=args.gen_batch)
    t4 = time.time()

    n = len(rows)
    out = []
    for i, r in enumerate(rows):
        X = r["target"]
        g_base, g_neg = greedy[i][0], greedy[n + i][0]
        s_base, s_neg = samples[i], samples[n + i]
        e = enc["neg"][i]
        row = {
            "prompt_id": r["prompt_id"], "category": r["category"], "question": r["question"],
            "target": X, "variants": sc_base[i][2],
            "variant_ntok": [len(v) for v in sc_base[i][3]],
            "variant_p_base": sc_base[i][1], "variant_p_neg": sc_neg[i][1],
            "p0": sc_base[i][0], "p1": sc_neg[i][0], "p0_bin": core.p0_bin(sc_base[i][0]),
            "t_star": tstar[i], "t_star_str": tok.decode([tstar[i]]), "tset": tset[i],
            "greedy_base": g_base, "greedy_neg": g_neg,
            "greedy_base_has_x": core.contains_word(X, g_base),
            "greedy_fail": core.contains_word(X, g_neg),
            "samples_base": s_base, "samples_neg": s_neg, "n_samples": len(s_neg),
            "base_x_rate": float(np.mean([core.contains_word(X, s) for s in s_base])),
            "viol_rate": float(np.mean([core.contains_word(X, s) for s in s_neg])),
            "span_tmf": e.span_text["tmf"], "span_nf": e.span_text["nf"],
            "span_q": e.span_text["q"], "span_ok": core.span_ok(e, X, r["question"]),
            "sample_seed": seed,
        }
        for c in core.CONDITIONS:
            for k, v in res[c][i].items():
                v = core.to_np(v)
                row[f"{k}_{c}"] = v.tolist() if v.ndim else v.item()
            row[f"top1_{c}"] = tok.decode([row[f"top1_{c}"]])
        for k in ("attn_tmf_base", "attn_nf_base", "attn_q_base"):
            row.pop(k, None)
        out.append(row)
    df = pd.DataFrame(out)
    df.attrs["timing"] = dict(score=t1 - t0, hooked=t2 - t1, greedy=t3 - t2, sample=t4 - t3)
    return df


def run_stage1(lm, prompts, args):
    folder = os.path.join(args.out, "measure")
    done = set(read_parts(folder, ["prompt_id"])["prompt_id"]) if os.path.isdir(folder) else set()
    todo = [r for r in prompts if r["prompt_id"] not in done]
    log(args.out, f"stage 1: {len(prompts)} prompts, {len(done)} already done, {len(todo)} to do")
    t_start, n_done = time.time(), 0
    for k in range(0, len(todo), args.chunk):
        rows = todo[k: k + args.chunk]
        tc = time.time()
        df = measure_chunk(lm, rows, args)
        path = write_part(df, folder)
        n_done += len(rows)
        el = time.time() - t_start
        rate = n_done / el
        tm = " ".join(f"{a}={b:.1f}s" for a, b in df.attrs["timing"].items())
        log(args.out, f"stage 1: {len(done) + n_done}/{len(prompts)} | chunk {len(rows)} in "
                      f"{time.time() - tc:.1f}s ({tm}) | {rate:.2f} prompts/s | "
                      f"ETA {(len(todo) - n_done) / rate / 60:.1f} min | {os.path.basename(path)}")
        if not df["span_ok"].all():
            log(args.out, f"WARNING span check failed for {list(df.loc[~df.span_ok, 'prompt_id'])}")


# ----------------------------------------------------------------------------- stage 2

def select_patch_subset(m: pd.DataFrame, per_bin: int, seed: int) -> list[str]:
    """Per P0 bin: up to per_bin/2 greedy failures and per_bin/2 successes, chosen at
    random (seeded); if one outcome is short, fill from the other."""
    rng = np.random.default_rng(seed)
    chosen = []
    for b in range(5):
        sub = m[m.p0_bin == b]
        fail = list(rng.permutation(sub.loc[sub.greedy_fail, "prompt_id"].to_numpy()))
        succ = list(rng.permutation(sub.loc[~sub.greedy_fail, "prompt_id"].to_numpy()))
        half = per_bin // 2
        nf = min(len(fail), max(half, per_bin - len(succ)))
        ns = min(len(succ), per_bin - nf)
        chosen += fail[:nf] + succ[:ns]
    return [str(x) for x in chosen]


@torch.no_grad()
def patch_chunk(lm: core.LM, rows: list[dict], args) -> pd.DataFrame:
    tok, L = lm.tok, lm.n_layers
    core.set_attn(lm, "eager")  # same attention kernel as stage 1, so 'none' matches P1
    base = [core.encode_prompt(tok, r["question"], r["target"], "base").ids for r in rows]
    neg = [core.encode_prompt(tok, r["question"], r["target"], "neg").ids for r in rows]
    cap, base_logits = core.hooked_forward(lm, base, want_attn=False)
    base_probs = torch.softmax(base_logits, -1)

    items = []  # (prompt i, kind, layer)
    for i in range(len(rows)):
        items.append((i, None, -1))
        items += [(i, "R", l) for l in range(L)] + [(i, "A", l) for l in range(L)]

    def fn(chunk):
        vec = torch.stack([cap["resid" if k == "R" else "attn"][i, max(l, 0)]
                           for i, k, l in chunk])
        _, logits = core.hooked_forward(
            lm, [neg[i] for i, _, _ in chunk], want_attn=False,
            patch={"kind": [k for _, k, _ in chunk], "layer": [l for _, _, l in chunk], "vec": vec})
        probs = torch.softmax(logits, -1)
        res = []
        for j, (i, k, l) in enumerate(chunk):
            ts = rows[i]["tset"]
            d = {"p_tok": probs[j, ts[0]].item(), "p_set": probs[j, ts].sum().item()}
            if k == "R" and l == L - 1:
                d["maxdiff"] = (probs[j] - base_probs[i]).abs().max().item()
            res.append(d)
        return res

    res = core.run_split(fn, items, args.patch_batch)
    out = []
    for i, r in enumerate(rows):
        mine = [(it, x) for it, x in zip(items, res) if it[0] == i]
        none = [x for (it, x) in mine if it[1] is None][0]
        R = [x for (it, x) in mine if it[1] == "R"]
        A = [x for (it, x) in mine if it[1] == "A"]
        ts = r["tset"]
        out.append({
            "prompt_id": r["prompt_id"], "category": r["category"], "p0": r["p0"],
            "p1": r["p1"], "p0_bin": r["p0_bin"], "greedy_fail": r["greedy_fail"],
            "viol_rate": r["viol_rate"],
            "p_none_tok": none["p_tok"], "p_none_set": none["p_set"],
            "p_base_tok": base_probs[i, ts[0]].item(), "p_base_set": base_probs[i, ts].sum().item(),
            "pR_tok": [x["p_tok"] for x in R], "pA_tok": [x["p_tok"] for x in A],
            "pR_set": [x["p_set"] for x in R], "pA_set": [x["p_set"] for x in A],
            "check_none_vs_stage1": abs(none["p_tok"] - r["p_t_neg"]),
            "check_Rlast_maxdiff": R[-1]["maxdiff"],
        })
    core.set_attn(lm, "sdpa")
    return pd.DataFrame(out)


def run_stage2(lm, args):
    m = read_parts(os.path.join(args.out, "measure"))
    subset_path = os.path.join(args.out, "patch_subset.json")
    if args.patch_ids:
        ids = [l.strip() for l in open(args.patch_ids) if l.strip()]
    elif os.path.exists(subset_path):
        ids = json.load(open(subset_path))
    else:
        ids = select_patch_subset(m, args.patch_per_bin, args.seed)
    json.dump(ids, open(subset_path, "w"), indent=0)
    folder = os.path.join(args.out, "patch")
    done = set(read_parts(folder, ["prompt_id"])["prompt_id"]) if os.path.isdir(folder) else set()
    byid = {r["prompt_id"]: r for r in m.to_dict("records")}
    missing = [i for i in ids if i not in byid]
    assert not missing, f"patch ids without stage-1 results: {missing[:5]}"
    todo = [byid[i] for i in ids if i not in done]
    for r in todo:
        r["tset"] = [int(x) for x in r["tset"]]
    counts = m[m.prompt_id.isin(ids)].groupby(["p0_bin", "greedy_fail"]).size().to_dict()
    log(args.out, f"stage 2: {len(ids)} prompts (by bin, greedy_fail: {counts}); "
                  f"{len(done)} done, {len(todo)} to do")
    t_start, n_done = time.time(), 0
    for k in range(0, len(todo), args.patch_chunk):
        rows = todo[k: k + args.patch_chunk]
        df = patch_chunk(lm, rows, args)
        write_part(df, folder)
        n_done += len(rows)
        rate = n_done / (time.time() - t_start)
        log(args.out, f"stage 2: {len(done) + n_done}/{len(ids)} | {rate:.2f} prompts/s | "
                      f"ETA {(len(todo) - n_done) / rate / 60:.1f} min | max none-vs-P1 diff "
                      f"{df.check_none_vs_stage1.max():.2e} | max R-last diff "
                      f"{df.check_Rlast_maxdiff.max():.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", help="auto = bfloat16 on CUDA, float32 elsewhere")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64, help="prompts per written part")
    ap.add_argument("--score-batch", type=int, default=256, help="sequences per P0/P1 forward")
    ap.add_argument("--fwd-batch", type=int, default=64, help="prompts per hooked forward")
    ap.add_argument("--gen-batch", type=int, default=1024, help="sequences per generate call")
    ap.add_argument("--patch-ids", default=None, help="file with one prompt_id per line")
    ap.add_argument("--patch-per-bin", type=int, default=50)
    ap.add_argument("--patch-chunk", type=int, default=10, help="prompts per patching part")
    ap.add_argument("--patch-batch", type=int, default=128, help="sequences per patch forward")
    ap.add_argument("--skip-patching", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    prompts = load_prompts(args.prompts)[: args.limit]
    t = time.time()
    lm = core.load_lm(args.model, args.device, args.dtype)
    log(args.out, f"loaded {args.model} on {lm.device} ({lm.n_layers} layers) in {time.time() - t:.0f}s")
    check_invariants(args.out, args, lm)
    run_stage1(lm, prompts, args)
    if not args.skip_patching:
        run_stage2(lm, args)
    log(args.out, "done")


if __name__ == "__main__":
    main()
