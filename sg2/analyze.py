"""Every number and figure in DESIGN_v2.md "Analysis", computed from saved run outputs only.

    python -m sg2.analyze --run runs/v2 [--out runs/v2/analysis] [--boot 1000]

Writes <out>/tables.json and PNG figures. Prompt-level outcome: a prompt "fails" when its
greedy negative completion contains X (greedy_fail); successes are the rest. Lens and
patching use the target-set probability (any surface variant's first token); DLA uses
the single target token t*. See sg2/README.md "Decisions".
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from sg2.run_experiment import read_parts  # noqa: E402

BINS = ["[0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1]"]
OUTCOMES = {"fail": True, "success": False}
COLORS = {"fail": "#c0392b", "success": "#2b6cb0"}


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ----------------------------------------------------------------------------- logistic

def fit_logistic(x, k, n, iters=100):
    """Binomial logistic regression logit(p) = a + b x on aggregated counts (equivalent to
    the sample-level Bernoulli fit). Newton-Raphson; returns (a, b)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (k - n * p)
        H = (X * (n * p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def bernoulli_ll(p, k, n):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))


def mcfadden(x, k, n, beta):
    p = 1 / (1 + np.exp(-(beta[0] + beta[1] * x)))
    p_null = k.sum() / n.sum()
    return 1 - bernoulli_ll(p, k, n) / bernoulli_ll(np.full_like(p, p_null), k, n)


def violation_vs_p0(m, boot, rng, out_dir):
    x = m.p0.to_numpy()
    n = m.n_samples.to_numpy().astype(float)
    k = np.round(m.viol_rate.to_numpy() * n)
    beta = fit_logistic(x, k, n)
    r2_mcf = mcfadden(x, k, n, beta)
    grid = np.linspace(0, 1, 101)
    curves, betas, r2s = [], [], []
    for _ in range(boot):  # resample prompts (clusters of 16 samples)
        idx = rng.integers(0, len(x), len(x))
        b = fit_logistic(x[idx], k[idx], n[idx])
        betas.append(b)
        r2s.append(mcfadden(x[idx], k[idx], n[idx], b))
        curves.append(1 / (1 + np.exp(-(b[0] + b[1] * grid))))
    betas, curves = np.array(betas), np.array(curves)
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)

    # Bin means (5 design bins): sample-level violation rate per bin vs the fitted curve
    # evaluated at the bin's mean P0. Labelled separately: it measures how well the curve
    # tracks the averages, not how well P0 predicts individual samples.
    bm = m.groupby("p0_bin").agg(p0_mean=("p0", "mean"), viol=("viol_rate", "mean"),
                                 n_prompts=("p0", "size"))
    bm["fit"] = 1 / (1 + np.exp(-(beta[0] + beta[1] * bm.p0_mean)))
    ss_res = float(((bm.viol - bm.fit) ** 2).sum())
    ss_tot = float(((bm.viol - bm.viol.mean()) ** 2).sum())
    r2_bins = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Bin-mean CIs by prompt bootstrap within bin.
    bin_ci = {}
    for b, g in m.groupby("p0_bin"):
        v = g.viol_rate.to_numpy()
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(boot)]
        bin_ci[int(b)] = np.percentile(bs, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(x, m.viol_rate, s=6, alpha=0.15, color="#555", label="prompts")
    ax.fill_between(grid, lo, hi, color="#2b6cb0", alpha=0.2, label="95% CI (prompt bootstrap)")
    ax.plot(grid, 1 / (1 + np.exp(-(beta[0] + beta[1] * grid))), color="#2b6cb0", lw=2,
            label=f"logistic fit (McFadden R²={r2_mcf:.3f})")
    err = np.array([[bm.viol[b] - bin_ci[b][0], bin_ci[b][1] - bm.viol[b]] for b in bm.index]).T
    ax.errorbar(bm.p0_mean, bm.viol, yerr=err, fmt="o", color="#c0392b", capsize=3,
                label=f"bin means (R² of fit to bin means={r2_bins:.3f})")
    ax.set_xlabel("P0 (baseline probability that the first word is X)")
    ax.set_ylabel("violation rate under the negative instruction")
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig1_violation_vs_p0.png"), dpi=200)
    plt.close(fig)

    return {
        "model": "logit(P(violation per sample)) = a + b * P0; fit on all negative samples",
        "n_prompts": len(x), "n_samples": int(n.sum()), "n_violations": int(k.sum()),
        "a": beta[0], "b": beta[1],
        "a_ci95": np.percentile(betas[:, 0], [2.5, 97.5]),
        "b_ci95": np.percentile(betas[:, 1], [2.5, 97.5]),
        "mcfadden_r2_sample_level": r2_mcf,
        "mcfadden_r2_ci95": np.percentile(r2s, [2.5, 97.5]),
        "r2_of_fit_to_5_bin_means": r2_bins,
        "bin_means": {int(b): {"p0_mean": r.p0_mean, "violation_rate": r.viol,
                               "ci95": bin_ci[int(b)], "fit": r.fit, "n_prompts": int(r.n_prompts)}
                      for b, r in bm.iterrows()},
        "curve_grid": grid, "curve_fit": 1 / (1 + np.exp(-(beta[0] + beta[1] * grid))),
        "curve_ci_lo": lo, "curve_ci_hi": hi,
    }


# ----------------------------------------------------------------------------- suppression

def suppression(m, boot, rng, out_dir):
    m = m.assign(supp=m.p0 - m.p1)
    per_bin, diffs, weights = {}, [], []
    for b in range(5):
        g = m[m.p0_bin == b]
        f, s = g[g.greedy_fail].supp.to_numpy(), g[~g.greedy_fail].supp.to_numpy()
        d = {"n_fail": len(f), "n_success": len(s),
             "mean_fail": f.mean() if len(f) else np.nan,
             "mean_success": s.mean() if len(s) else np.nan}
        if len(f) and len(s):
            d["diff_fail_minus_success"] = d["mean_fail"] - d["mean_success"]
            d["mannwhitney_p"] = stats.mannwhitneyu(f, s).pvalue if len(f) > 1 or len(s) > 1 else np.nan
            diffs.append(d["diff_fail_minus_success"])
            weights.append(len(f) * len(s) / (len(f) + len(s)))
        per_bin[BINS[b]] = d

    def pooled(df):
        ds, ws = [], []
        for b in range(5):
            g = df[df.p0_bin == b]
            f, s = g[g.greedy_fail].supp, g[~g.greedy_fail].supp
            if len(f) and len(s):
                ds.append(f.mean() - s.mean()); ws.append(len(f) * len(s) / (len(f) + len(s)))
        return np.average(ds, weights=ws) if ds else np.nan

    est = pooled(m)
    bs = []
    for _ in range(boot):  # resample prompts within each bin
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for _, g in m.groupby("p0_bin")]
        bs.append(pooled(pd.concat(parts)))
    bs = np.array(bs)
    bs = bs[np.isfinite(bs)]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    w = 0.38
    for j, (name, val) in enumerate(OUTCOMES.items()):
        means, errs = [], []
        for b in range(5):
            v = m[(m.p0_bin == b) & (m.greedy_fail == val)].supp.to_numpy()
            means.append(v.mean() if len(v) else np.nan)
            errs.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(np.arange(5) + (j - 0.5) * w, means, w, yerr=errs, color=COLORS[name], capsize=2,
               label=f"{name} (n={int((m.greedy_fail == val).sum())})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(5), BINS); ax.set_xlabel("P0 bin")
    ax.set_ylabel("suppression P0 − P1 (mean ± SE)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig2_suppression_by_bin.png"), dpi=200)
    plt.close(fig)
    return {"definition": "suppression = P0 - P1; outcome = greedy failure; compared within P0 bins",
            "per_bin": per_bin,
            "pooled_within_bin_diff_fail_minus_success": est,
            "pooled_ci95": np.percentile(bs, [2.5, 97.5]) if len(bs) else [np.nan, np.nan],
            "pooled_weights": "per-bin n_fail*n_success/(n_fail+n_success)"}


# ----------------------------------------------------------------------------- signatures

def signatures(m, boot, rng):
    tmf = np.array([np.mean(v) for v in m.attn_tmf_neg])
    nf = np.array([np.mean(v) for v in m.attn_nf_neg])
    sigs = {"tmf_gt_nf (layer-mean attention on target mention > on 'Do not')": tmf > nf,
            "p1_gt_p0": (m.p1 > m.p0).to_numpy()}
    fail = m.greedy_fail.to_numpy()
    out = {}
    for name, s in sigs.items():
        a, b = int((s & fail).sum()), int((~s & fail).sum())
        c, d = int((s & ~fail).sum()), int((~s & ~fail).sum())
        rf = a / (a + b) if a + b else np.nan
        rs = c / (c + d) if c + d else np.nan
        bs = []
        for _ in range(boot):
            idx = rng.integers(0, len(s), len(s))
            ss, ff = s[idx], fail[idx]
            if ff.any() and (~ff).any():
                bs.append(ss[ff].mean() - ss[~ff].mean())
        per_bin = {}
        for bb in range(5):
            sel = (m.p0_bin == bb).to_numpy()
            per_bin[BINS[bb]] = {
                "rate_fail": s[sel & fail].mean() if (sel & fail).any() else np.nan,
                "rate_success": s[sel & ~fail].mean() if (sel & ~fail).any() else np.nan,
                "n_fail": int((sel & fail).sum()), "n_success": int((sel & ~fail).sum())}
        out[name] = {"rate_among_failures": rf, "rate_among_successes": rs,
                     "difference": rf - rs,
                     "difference_ci95_bootstrap": np.percentile(bs, [2.5, 97.5]) if bs else None,
                     "fisher_exact_p": stats.fisher_exact([[a, b], [c, d]]).pvalue,
                     "table_[[sig&fail, ~sig&fail],[sig&success, ~sig&success]]": [[a, b], [c, d]],
                     "per_p0_bin": per_bin}
    return out


# ----------------------------------------------------------------------------- curves

def curve_stats(arrs):
    a = np.array([np.asarray(v, float) for v in arrs])
    if len(a) == 0:
        return None
    se = a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros(a.shape[1])
    return {"n": len(a), "mean": a.mean(0), "se": se}


def curves_by_bin(m, out_dir):
    metrics = {
        "lens_set_base": "lens P(target set), baseline", "lens_set_neg": "lens P(target set), negative",
        "lens_tok_neg": "lens P(t*), negative",
        "dla_attn_neg": "DLA attention → t*, negative", "dla_mlp_neg": "DLA MLP → t*, negative",
        "dla_attn_base": "DLA attention → t*, baseline", "dla_mlp_base": "DLA MLP → t*, baseline",
        "dla_attn_c_neg": "centred DLA attention, negative", "dla_mlp_c_neg": "centred DLA MLP, negative",
        "attn_tmf_neg": "attention on target mention", "attn_nf_neg": "attention on 'Do not'",
        "attn_q_neg": "attention on question",
    }
    tables = {}
    for key in metrics:
        tables[key] = {BINS[b]: {o: curve_stats(m[(m.p0_bin == b) & (m.greedy_fail == v)][key])
                                 for o, v in OUTCOMES.items()} for b in range(5)}
    figs = {"fig3_logit_lens_by_bin.png": ["lens_set_base", "lens_set_neg"],
            "fig4_dla_by_bin.png": ["dla_attn_neg", "dla_mlp_neg", "dla_attn_base", "dla_mlp_base"],
            "fig5_attention_by_bin.png": ["attn_tmf_neg", "attn_nf_neg", "attn_q_neg"]}
    for fname, keys in figs.items():
        fig, axes = plt.subplots(len(keys), 5, figsize=(15, 2.6 * len(keys)), sharex=True,
                                 sharey="row", squeeze=False)
        for r, key in enumerate(keys):
            for b in range(5):
                ax = axes[r, b]
                for o in OUTCOMES:
                    st = tables[key][BINS[b]][o]
                    if st is None:
                        continue
                    xs = np.arange(len(st["mean"]))
                    ax.plot(xs, st["mean"], color=COLORS[o], label=f"{o} (n={st['n']})")
                    ax.fill_between(xs, st["mean"] - st["se"], st["mean"] + st["se"],
                                    color=COLORS[o], alpha=0.2)
                if r == 0:
                    ax.set_title(f"P0 bin {BINS[b]}", fontsize=9)
                if b == 0:
                    ax.set_ylabel(metrics[key], fontsize=8)
                if r == len(keys) - 1:
                    ax.set_xlabel("layer")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=6, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)
    return tables


def patching(p, boot, rng, out_dir):
    if p is None or len(p) == 0:
        return None
    res = {"effect_definition": "P_patched(target set, next token) - P_unpatched(target set) "
                                "on the negative prompt; types reported separately, never averaged",
           "n_prompts": len(p), "checks": {
               "max_abs_none_vs_stage1_p_tstar": float(p.check_none_vs_stage1.max()),
               "max_abs_Rlast_vs_baseline_distribution": float(p.check_Rlast_maxdiff.max())}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for j, typ in enumerate(["R", "A"]):
        res[typ] = {}
        for o, v in OUTCOMES.items():
            g = p[p.greedy_fail == v]
            if not len(g):
                continue
            for metric in ("set", "tok"):
                eff = np.array([np.asarray(x) - n0 for x, n0 in
                                zip(g[f"p{typ}_{metric}"], g[f"p_none_{metric}"])])
                bs = np.array([eff[rng.integers(0, len(eff), len(eff))].mean(0) for _ in range(boot)])
                res[typ].setdefault(o, {})[metric] = {
                    "n": len(eff), "mean": eff.mean(0), "ci95_lo": np.percentile(bs, 2.5, 0),
                    "ci95_hi": np.percentile(bs, 97.5, 0),
                    "by_p0_bin": {BINS[b]: eff[(g.p0_bin == b).to_numpy()].mean(0)
                                  for b in range(5) if (g.p0_bin == b).any()}}
            st = res[typ][o]["set"]
            xs = np.arange(len(st["mean"]))
            axes[j].plot(xs, st["mean"], color=COLORS[o], label=f"{o} (n={st['n']})")
            axes[j].fill_between(xs, st["ci95_lo"], st["ci95_hi"], color=COLORS[o], alpha=0.2)
        axes[j].axhline(0, color="k", lw=0.6)
        axes[j].set_title({"R": "residual stream patched (type R)",
                           "A": "attention output patched (type A)"}[typ], fontsize=10)
        axes[j].set_xlabel("layer patched (baseline → negative, decision position)")
        axes[j].legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("change in P(target set)\n(mean, 95% prompt-bootstrap CI)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig6_patching.png"), dpi=200)
    plt.close(fig)
    return res


def descriptive(m):
    agree = pd.crosstab(m.greedy_fail, m.viol_rate >= 0.5)
    return {
        "n_prompts": len(m), "n_layers": len(m.lens_set_neg.iloc[0]),
        "n_by_category": m.category.value_counts().to_dict(),
        "n_by_p0_bin": {BINS[b]: int((m.p0_bin == b).sum()) for b in range(5)},
        "max_p0": m.p0.max(), "max_p1": m.p1.max(),
        "span_check_failures": int((~m.span_ok).sum()),
        "share_preferred_variant_single_token": float(m.target_single_token.mean()),
        "sample_violation_rate": m.viol_rate.mean(),
        "greedy_failure_rate": m.greedy_fail.mean(),
        "baseline_sample_rate_of_x": m.base_x_rate.mean(),
        "baseline_greedy_has_x_rate": m.greedy_base_has_x.mean(),
        "by_category": {c: {"n": len(g), "violation_rate": g.viol_rate.mean(),
                            "greedy_failure_rate": g.greedy_fail.mean(),
                            "mean_p0": g.p0.mean(), "mean_p1": g.p1.mean()}
                        for c, g in m.groupby("category")},
        "violation_rate_by_category_and_bin": {
            c: {BINS[int(b)]: gg.viol_rate.mean() for b, gg in g.groupby("p0_bin")}
            for c, g in m.groupby("category")},
        "greedy_fail_vs_sample_majority_crosstab": {
            f"greedy_fail={i}": {f"viol_rate>=0.5={j}": int(agree.loc[i, j])
                                 for j in agree.columns} for i in agree.index},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "analysis")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    m = read_parts(os.path.join(args.run, "measure"))
    if "target_single_token" not in m:  # runs written before the flag existed
        m["target_single_token"] = [nt[int(np.argmax(np.add(pb, pn)))] == 1 for nt, pb, pn in
                                    zip(m.variant_ntok, m.variant_p_base, m.variant_p_neg)]
    p = read_parts(os.path.join(args.run, "patch")) if os.path.isdir(os.path.join(args.run, "patch")) else None
    # Every mechanistic result twice: all prompts, and prompts whose target's preferred
    # spelling is a single Qwen token (there first-token measures are exact, not a proxy).
    single = m.target_single_token.to_numpy(bool)
    subsets = {"all_prompts": (m, p, out),
               "single_token_targets": (m[single], None if p is None else
                                        p[p.prompt_id.isin(m.prompt_id[single])],
                                        os.path.join(out, "single_token"))}
    mech = {}
    for name, (mm, pp, od) in subsets.items():
        os.makedirs(od, exist_ok=True)
        if len(mm) < 2:
            mech[name] = {"n_prompts": len(mm), "skipped": "fewer than 2 prompts"}
            continue
        mech[name] = {
            "n_prompts": len(mm), "figures_dir": os.path.relpath(od, out),
            "suppression_by_outcome_within_bins": suppression(mm, args.boot, rng, od),
            "signatures": signatures(mm, args.boot, rng),
            "curves_by_p0_bin_and_outcome": curves_by_bin(mm, od),
            "patching": patching(pp, args.boot, rng, od),
        }
    tables = {
        "descriptive": descriptive(m),
        "violation_vs_p0": violation_vs_p0(m, args.boot, rng, out),
        "mechanistic": mech,
        "outcome_definition": "fail = greedy negative completion contains X as a whole word",
        "single_token_definition": "the variant of X with the highest P0+P1 is one Qwen token",
        "bootstrap": {"draws": args.boot, "unit": "prompt", "seed": args.seed},
    }
    json.dump(clean(tables), open(os.path.join(out, "tables.json"), "w"), indent=1)
    d = tables["descriptive"]
    v = tables["violation_vs_p0"]
    print(f"{d['n_prompts']} prompts; violation rate {d['sample_violation_rate']:.3f}; greedy failure "
          f"{d['greedy_failure_rate']:.3f}; logistic b={v['b']:.2f} CI {v['b_ci95']}; McFadden R2 "
          f"{v['mcfadden_r2_sample_level']:.3f}; R2 bin means {v['r2_of_fit_to_5_bin_means']:.3f}")
    print(f"wrote {out}/tables.json and figures")


if __name__ == "__main__":
    main()
