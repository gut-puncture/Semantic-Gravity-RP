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
            plain.append(torch.softmax(lm.model(input_ids=ids, use_cache=False).logits[0, -1].float(), -1))
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
    if args.run_dir:
        # The run's own patching output: it must exist, cover exactly the ids in
        # patch_subset.json (each once), hold finite values on every row, and its unpatched
        # target-set probability must agree with stage 1's final-layer value (negative prompt).
        pdir = os.path.join(args.run_dir, "patch")
        subset_path = os.path.join(args.run_dir, "patch_subset.json")
        if not (os.path.isdir(pdir) and os.path.exists(subset_path)):
            report("c_patching_in_run", False, error="no patch folder or patch_subset.json in the run")
        else:
            pr = read_parts(pdir)
            want = json.load(open(subset_path))
            got = list(pr.prompt_id)
            ids_ok = len(got) == len(set(got)) and set(got) == set(want) and len(want) == len(set(want))
            ms = read_parts(os.path.join(args.run_dir, "measure"), ["prompt_id", "lens_set_neg"])
            s1 = {p: float(v[-1]) for p, v in zip(ms.prompt_id, ms.lens_set_neg)}
            set_diff = np.array([abs(a - s1.get(p, np.nan)) for p, a in zip(pr.prompt_id, pr.p_none_set)])
            vals = np.concatenate([pr.check_none_vs_stage1.to_numpy(float),
                                   pr.check_Rlast_maxdiff.to_numpy(float), set_diff,
                                   *[np.asarray(v, float) for c in ("pR_tok", "pA_tok", "pR_set", "pA_set")
                                     for v in pr[c]]])
            finite = bool(np.isfinite(vals).all())
            tol_r = 1e-4 if fp32 else 2e-2
            report("c_patching_in_run", ids_ok and finite and len(pr) > 0
                   and pr.check_none_vs_stage1.max() < tol_r and pr.check_Rlast_maxdiff.max() < tol_r
                   and set_diff.max() < tol_r,
                   n_prompts=len(pr), n_expected=len(want), ids_match=ids_ok, all_finite=finite,
                   max_none_vs_P1_tok=float(pr.check_none_vs_stage1.max()),
                   max_none_set_vs_stage1_set=float(set_diff.max()) if len(set_diff) else float("nan"),
                   max_Rlast_vs_base=float(pr.check_Rlast_maxdiff.max()))
    core.set_attn(lm, "sdpa")

    # (b) P0 <= 1 for every prompt and condition; brute-force check on the 3 prompts with
    # P0 closest to 0.5 (most informative).
    sc = core.score_variants(lm, seqs, [r["target"] for r in rows] * 2, 64)
    pmax = max(s[0] for s in sc)
    order = sorted(range(len(rows)), key=lambda i: abs(sc[i][0] - 0.5))[:3]
    zs, details = [], []
    # Draws per generate call: capped by GPU memory (256 is the production cap below 40 GB),
    # and halved on out-of-memory. The number of draws is unchanged.
    k_cap = 500
    if lm.device.type == "cuda" and torch.cuda.get_device_properties(lm.device).total_memory < 40 * 2**30:
        k_cap = 256
    for i in order:
        p0 = sc[i][0]
        # The reference is built here from the design text, not taken from the scorer under
        # test: {X, x, X.capitalize(), X.upper()} deduplicated, each with and without a leading
        # space; the boundary rule is applied to the decoded text of each sampled next token.
        X = rows[i]["target"]
        forms = []
        for f in (X, X.lower(), X[:1].upper() + X[1:].lower(), X.upper()):
            if f not in forms:
                forms.append(f)
        vs = forms + [" " + f for f in forms]
        vids = [tok.encode(v, add_special_tokens=False) for v in vs]
        starts_word_char = {}
        maxlen = max(len(v) for v in vids) + 1
        cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                               repetition_penalty=1.0, max_new_tokens=maxlen,
                               eos_token_id=lm.eos_ids, pad_token_id=lm.pad_id)
        ids = torch.tensor([base[i]], device=lm.device)
        hits, text_hits, n = 0, 0, 0
        with torch.no_grad():
            while n < args.n_brute:
                k = min(k_cap, args.n_brute - n)
                try:
                    out = lm.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                            generation_config=cfg, num_return_sequences=k)
                except Exception as e:  # noqa: BLE001
                    if not core._is_oom(e) or k_cap == 1:
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    k_cap = max(1, k_cap // 2)
                    print(f"[oom] brute-force batch reduced to {k_cap}", flush=True)
                    continue
                for g in out[:, ids.shape[1]:].tolist():
                    for v in vids:
                        if g[: len(v)] == v and len(g) > len(v):
                            t_next = g[len(v)]
                            if t_next not in starts_word_char:
                                starts_word_char[t_next] = tok.decode([t_next])[:1].isalnum()
                            if not starts_word_char[t_next]:
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
