"""Golden / integration checks for the v2 pipeline, run against the real model (no mocks).

    python -m sg2.checks --model Qwen/Qwen2.5-7B-Instruct --prompts sg2/test_prompts.jsonl \
        --out runs/checks [--run-dir runs/v2]

(a) logit lens at the last layer equals the model's real output distribution;
(b) P0 <= 1, and P0 equals a brute-force sampling estimate within sampling error;
(c) patching nothing reproduces the unpatched run exactly; patching the last-layer
    residual reproduces the baseline distribution;
(d) TMF / NF / question token spans decode back to the expected text;
(e) left padding: greedy output alone == inside a padded batch;
(f) DLA terms plus the embedding term sum to the real target logit (linearity).
Exit code 1 if any check fails. Results also written to <out>/checks.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import GenerationConfig

from sg2 import core
from sg2.run_experiment import load_prompts, read_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "test_prompts.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--n-brute", type=int, default=3000, help="samples per prompt for (b)")
    ap.add_argument("--run-dir", default=None, help="also report in-run patching checks")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    lm = core.load_lm(args.model, args.device, args.dtype)
    fp32 = next(lm.model.parameters()).dtype == torch.float32
    rows = load_prompts(args.prompts)
    tok, L = lm.tok, lm.n_layers
    results = {}

    def report(name, ok, **nums):
        results[name] = {"pass": bool(ok), **nums}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: " +
              ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in nums.items()),
              flush=True)

    enc = {c: [core.encode_prompt(tok, r["question"], r["target"], c) for r in rows]
           for c in core.CONDITIONS}
    seqs = [e.ids for e in enc["base"]] + [e.ids for e in enc["neg"]]

    # (a) logit lens, last layer, from the batched hooked pass vs a plain unbatched forward.
    core.set_attn(lm, "eager")
    cap, logits = core.hooked_forward(lm, seqs, want_attn=True)
    lens_last = core.lens_dist(lm, cap["resid"][:, -1])
    same_pass = (lens_last - torch.softmax(logits, -1)).abs().max().item()
    plain = []
    with torch.no_grad():
        for s in seqs:
            ids = torch.tensor([s], device=lm.device)
            plain.append(torch.softmax(lm.model(input_ids=ids).logits[0, -1].float(), -1))
    indep = (lens_last - torch.stack(plain)).abs().max().item()
    tol_a = 1e-3 if fp32 else 2e-2
    report("a_logit_lens_last_layer", same_pass < 1e-6 and indep < tol_a,
           max_abs_diff_same_pass=same_pass, max_abs_diff_vs_plain_unbatched=indep, tol=tol_a,
           n_dists=len(seqs))

    # (f) DLA linearity on the same pass.
    tstar = torch.tensor([tok.encode(r["target"], add_special_tokens=False)[0] for r in rows] * 2,
                         device=lm.device)
    d = core.dla(lm, cap, tstar)
    recon = d["dla_embed"] + d["dla_attn"].sum(1) + d["dla_mlp"].sum(1)
    real = logits.gather(1, tstar[:, None]).squeeze(1).cpu()
    rel = ((recon - real).abs() / real.abs().clamp_min(1)).max().item()
    report("f_dla_sums_to_logit", rel < (1e-4 if fp32 else 2e-2), max_rel_diff=rel)

    # (d) spans.
    bad = [(r["prompt_id"], e.span_text) for r, e in zip(rows, enc["neg"])
           if not core.span_ok(e, r["target"], r["question"])]
    att = core.attention_masses(cap["attw"][len(rows):], [e.spans for e in enc["neg"]])
    ex = enc["neg"][0].span_text
    report("d_token_spans", not bad, n_prompts=len(rows), n_bad=len(bad),
           example=json.dumps(ex), attn_nan=int(sum(torch.isnan(v).sum().item() for v in att.values())))
    for pid, t in bad:
        print("   bad span", pid, t)

    # (c) patching: nothing patched == unpatched (same batch), and last-layer residual
    # patch == baseline distribution.
    neg = [e.ids for e in enc["neg"]]
    base = [e.ids for e in enc["base"]]
    capb, logits_b = core.hooked_forward(lm, base, want_attn=False)
    _, logits_n = core.hooked_forward(lm, neg, want_attn=False)
    B = len(rows)
    zero = torch.zeros(B, capb["resid"].shape[-1], device=lm.device)
    _, l_none = core.hooked_forward(lm, neg, False, patch={"kind": [None] * B, "layer": [0] * B, "vec": zero})
    none_diff = (torch.softmax(l_none, -1) - torch.softmax(logits_n, -1)).abs().max().item()
    _, l_R = core.hooked_forward(lm, neg, False,
                                 patch={"kind": ["R"] * B, "layer": [L - 1] * B, "vec": capb["resid"][:, -1]})
    r_diff = (torch.softmax(l_R, -1) - torch.softmax(logits_b, -1)).abs().max().item()
    # Layer-0 attention patch must change something (guards against a no-op hook).
    _, l_A0 = core.hooked_forward(lm, neg, False,
                                  patch={"kind": ["A"] * B, "layer": [0] * B, "vec": capb["attn"][:, 0]})
    a_change = (torch.softmax(l_A0, -1) - torch.softmax(logits_n, -1)).abs().max().item()
    tol_c = 1e-5 if fp32 else 1e-2
    report("c_patching", none_diff == 0.0 and r_diff < tol_c and a_change > 1e-4,
           none_patch_max_abs_diff=none_diff, Rlast_vs_baseline_max_abs_diff=r_diff,
           A0_patch_effect=a_change, tol=tol_c)
    if args.run_dir and os.path.isdir(os.path.join(args.run_dir, "patch")):
        pr = read_parts(os.path.join(args.run_dir, "patch"))
        report("c_patching_in_run", pr.check_none_vs_stage1.max() < (1e-4 if fp32 else 2e-2)
               and pr.check_Rlast_maxdiff.max() < (1e-4 if fp32 else 2e-2),
               n_prompts=len(pr), max_none_vs_P1_tok=float(pr.check_none_vs_stage1.max()),
               max_Rlast_vs_base=float(pr.check_Rlast_maxdiff.max()))
    core.set_attn(lm, "sdpa")

    # (b) P0 <= 1 for every prompt and condition; brute-force check on the 3 prompts with
    # P0 closest to 0.5 (most informative).
    sc = core.score_variants(lm, seqs, [r["target"] for r in rows] * 2, 64)
    pmax = max(s[0] for s in sc)
    order = sorted(range(len(rows)), key=lambda i: abs(sc[i][0] - 0.5))[:3]
    zs, details = [], []
    for i in order:
        p0, _, vs, vids = sc[i]
        maxlen = max(len(v) for v in vids) + 1
        cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                               repetition_penalty=1.0, max_new_tokens=maxlen,
                               eos_token_id=lm.eos_ids, pad_token_id=lm.pad_id)
        ids = torch.tensor([base[i]], device=lm.device)
        hits, text_hits, n = 0, 0, 0
        with torch.no_grad():
            while n < args.n_brute:
                k = min(500, args.n_brute - n)
                out = lm.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        generation_config=cfg, num_return_sequences=k)
                for g in out[:, ids.shape[1]:].tolist():
                    for v in vids:
                        if g[: len(v)] == v and len(g) > len(v) and not lm.starts_alnum[g[len(v)]]:
                            hits += 1
                            break
                    text = tok.decode(g, skip_special_tokens=True)
                    text_hits += any(text == v or (text.startswith(v) and not text[len(v)].isalnum())
                                     for v in vs)
                n += k
        f = hits / n
        se = math.sqrt(max(p0 * (1 - p0), 1e-12) / n)
        z = (f - p0) / se
        zs.append(abs(z))
        details.append({"prompt_id": rows[i]["prompt_id"], "P0": p0, "freq_canonical": f,
                        "freq_any_tokenisation": text_hits / n, "z": z, "n": n})
        print(f"   {rows[i]['prompt_id']}: P0={p0:.4f} brute={f:.4f} (any tokenisation "
              f"{text_hits / n:.4f}) n={n} z={z:+.2f}")
    report("b_p0_bounded_and_brute_force", pmax <= 1 + 1e-4 and max(zs) < 4,
           max_P=pmax, max_abs_z=max(zs), prompts=json.dumps(details))

    # (e) left padding: greedy alone vs in a padded batch (lengths differ across prompts).
    t = time.time()
    batched = core.generate(lm, seqs, do_sample=False, n=1, max_new_tokens=8, seq_batch=len(seqs))
    alone = [core.generate(lm, [s], do_sample=False, n=1, max_new_tokens=8)[0] for s in seqs]
    mism = [(i, a[0], b[0]) for i, (a, b) in enumerate(zip(alone, batched)) if a != b]
    lens = [len(s) for s in seqs]
    report("e_left_padding_greedy", not mism, n=len(seqs), n_mismatch=len(mism),
           prompt_len_range=f"{min(lens)}-{max(lens)}")
    for m in mism:
        print("   mismatch", m)

    json.dump(results, open(os.path.join(args.out, "checks.json"), "w"), indent=2)
    ok = all(r["pass"] for r in results.values())
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
