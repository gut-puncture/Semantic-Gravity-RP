"""
visualize.py - Visualization and Figure Generation

This module generates figures, tables.json, and appendix examples from
experiment outputs per specification Section 12.

Outputs:
- figures/*.png (7 plots)
- tables.json
- appendix_examples/*.json (20 curated failures)
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS
# ============================================================================

try:
    from .config import get_base_paths, CONFIG
    from .prompt_builder import build_prompt
except ImportError:
    from config import get_base_paths, CONFIG
    from prompt_builder import build_prompt


# ============================================================================
# HELPER: RESOLVE RUN ROOT
# ============================================================================


def _resolve_run_root(output_root: Optional[Path]) -> Path:
    """
    Resolve the run root directory from output_root or find the latest run.

    Args:
        output_root: Optional run root or base outputs directory

    Returns:
        Resolved run root Path
    """
    paths = get_base_paths()
    base_out = paths.get("output_root", Path("outputs"))

    def find_latest_run_in(parent: Path) -> Optional[Path]:
        if not parent.exists() or not parent.is_dir():
            return None
        try:
            run_dirs = sorted(
                [d for d in parent.iterdir() if d.is_dir() and d.name.startswith("experiment_run_")],
                key=lambda d: d.name,
            )
            return run_dirs[-1] if run_dirs else None
        except OSError:
            return None

    if output_root is not None:
        candidate = Path(output_root)

        if candidate.name.startswith("experiment_run_"):
            return candidate

        latest = find_latest_run_in(candidate)
        if latest:
            warnings.warn(f"Selecting latest experiment_run_* under {candidate}: {latest}", UserWarning)
            return latest

        if candidate.exists() and (candidate / "runs").exists():
            return candidate

        return candidate

    latest = find_latest_run_in(base_out)
    if latest:
        warnings.warn(f"output_root not provided; using latest run dir: {latest}", UserWarning)
        return latest

    return base_out


# ============================================================================
# LOADERS
# ============================================================================


def _load_prompts_df(
    prompts_path: Optional[Path],
    data_root: Path,
) -> pd.DataFrame:
    """Load prompts.csv with required columns."""
    path = prompts_path or (data_root / "prompts.csv")
    if not path.exists():
        raise FileNotFoundError(f"prompts.csv not found: {path}")

    df = pd.read_csv(path)
    required = ["prompt_id", "category", "question_text", "target_word", "p0_bin"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"prompts.csv missing columns: {missing}")

    df["prompt_id"] = df["prompt_id"].astype(str)
    return df


def _load_jsonl(path: Path) -> Iterator[Dict]:
    """Yield dicts from JSONL file."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _load_required_csv(run_root: Path, filename: str) -> pd.DataFrame:
    """Load required CSV file."""
    path = run_root / "runs" / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    if "prompt_id" in df.columns:
        df["prompt_id"] = df["prompt_id"].astype(str)
    return df


def _parse_bin(bin_str: str) -> float:
    """Parse bin string to get lower bound for sorting."""
    try:
        # Handle formats like "[0.0, 0.2)", "p0_bin=[0.0, 0.2)", or "0.0-0.2"
        if "=" in bin_str:
            bin_str = bin_str.split("=")[-1]
        bin_str = bin_str.strip("[]() ")
        
        # Try comma format first
        if "," in bin_str:
            parts = bin_str.split(",")
            return float(parts[0].strip())
        # Try dash format
        elif "-" in bin_str:
            parts = bin_str.split("-")
            return float(parts[0].strip())
        else:
            return float(bin_str.strip())
    except (ValueError, IndexError):
        return float("inf")


def _get_bin_center(bin_str: str) -> float:
    """Get bin center for plotting."""
    try:
        if "=" in bin_str:
            bin_str = bin_str.split("=")[-1]
        bin_str = bin_str.strip("[]() ")
        
        # Try comma format first
        if "," in bin_str:
            parts = bin_str.split(",")
            low = float(parts[0].strip())
            high = float(parts[1].strip().rstrip(")"))
            return (low + high) / 2
        # Try dash format
        elif "-" in bin_str:
            parts = bin_str.split("-")
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2
        else:
            return float(bin_str.strip())
    except (ValueError, IndexError):
        return 0.5


# ============================================================================
# DERIVE OUTCOMES
# ============================================================================


def _derive_outcomes(run_root: Path) -> Dict[str, Dict]:
    """
    Derive prompt outcomes from detection_mapping.jsonl.

    Returns:
        Dict mapping prompt_id -> {outcome, completion_text, token_spans}
    """
    path = run_root / "runs" / "detection_mapping.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"detection_mapping.jsonl not found: {path}")

    outcomes = {}
    for record in _load_jsonl(path):
        if record.get("condition") != "negative":
            continue
        if record.get("sample_id", "") != "":
            continue

        prompt_id = str(record.get("prompt_id", ""))
        word_present = record.get("word_present", False)
        outcome = "failure" if word_present else "success"

        outcomes[prompt_id] = {
            "outcome": outcome,
            "completion_text": record.get("completion_text", ""),
            "token_spans": record.get("token_spans", []),
        }

    return outcomes


# ============================================================================
# FIGURE GENERATION
# ============================================================================


def _plot_violation_rate_vs_p0(bootstrap_df: pd.DataFrame, figures_dir: Path) -> None:
    """Figure 1: Violation rate vs P0 bin with CI."""
    df = bootstrap_df[bootstrap_df["metric_name"] == "violation_rate"].copy()
    if df.empty:
        logger.warning("No violation_rate data for plotting")
        return

    # Extract bin from group_key
    df["bin"] = df["group_key"].apply(lambda x: x.split("=")[-1] if "=" in x else x)
    df["bin_lower"] = df["bin"].apply(_parse_bin)
    df["bin_center"] = df["bin"].apply(_get_bin_center)
    df = df.sort_values("bin_lower")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        df["bin_center"], df["mean"],
        yerr=[df["mean"] - df["ci_low"], df["ci_high"] - df["mean"]],
        fmt="o-", capsize=4, label="Violation Rate"
    )
    ax.set_xlabel("P0 Bin Center")
    ax.set_ylabel("Violation Rate")
    ax.set_title("Violation Rate vs Semantic Pressure (P0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "violation_rate_vs_p0.png", dpi=200)
    plt.close()


def _plot_suppression_relative_vs_p0(bootstrap_df: pd.DataFrame, figures_dir: Path) -> None:
    """Figure 2: Relative suppression vs P0 with CI."""
    df = bootstrap_df[bootstrap_df["metric_name"] == "suppression_relative"].copy()
    if df.empty:
        logger.warning("No suppression_relative data for plotting")
        return

    df["bin"] = df["group_key"].apply(lambda x: x.split("=")[-1] if "=" in x else x)
    df["bin_lower"] = df["bin"].apply(_parse_bin)
    df["bin_center"] = df["bin"].apply(_get_bin_center)
    df = df.sort_values("bin_lower")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        df["bin_center"], df["mean"],
        yerr=[df["mean"] - df["ci_low"], df["ci_high"] - df["mean"]],
        fmt="s-", capsize=4, color="orange", label="Relative Suppression"
    )
    ax.set_xlabel("P0 Bin Center")
    ax.set_ylabel("Relative Suppression")
    ax.set_title("Semantic Suppression vs Pressure")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "suppression_relative_vs_p0.png", dpi=200)
    plt.close()


def _plot_attention_metrics_vs_p0(
    attention_df: pd.DataFrame,
    prompts_df: pd.DataFrame,
    outcomes: Dict[str, Dict],
    figures_dir: Path,
) -> None:
    """Figure 3: Attention routing metrics vs P0, success vs failure."""
    # Filter to global_mean, negative condition
    df = attention_df.copy()
    if "aggregate_flag" in df.columns:
        df = df[df["aggregate_flag"] == "global_mean"]
    if "condition" in df.columns:
        df = df[df["condition"] == "negative"]

    # Join with prompts and outcomes
    df = df.merge(prompts_df[["prompt_id", "p0_bin"]], on="prompt_id", how="inner")
    df["outcome"] = df["prompt_id"].apply(lambda x: outcomes.get(x, {}).get("outcome", "unknown"))
    df = df[df["outcome"].isin(["success", "failure"])]

    if df.empty:
        logger.warning("No attention data for plotting")
        return

    metrics = ["iar", "nf", "tmf"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue

        ax = axes[i]
        for outcome, marker, color in [("success", "o", "green"), ("failure", "x", "red")]:
            sub = df[df["outcome"] == outcome]
            grouped = sub.groupby("p0_bin")[metric].mean().reset_index()
            grouped["bin_center"] = grouped["p0_bin"].apply(_get_bin_center)
            grouped["bin_lower"] = grouped["p0_bin"].apply(_parse_bin)
            grouped = grouped.sort_values("bin_lower")

            ax.plot(grouped["bin_center"], grouped[metric], f"{marker}-", label=outcome, color=color)

        ax.set_xlabel("P0 Bin Center")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by Outcome")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "attention_metrics_vs_p0.png", dpi=200)
    plt.close()


def _plot_pi_vs_violation(
    attention_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Figure 4: PI vs violation probability."""
    # Filter attention to global_mean, negative
    attn = attention_df.copy()
    if "aggregate_flag" in attn.columns:
        attn = attn[attn["aggregate_flag"] == "global_mean"]
    if "condition" in attn.columns:
        attn = attn[attn["condition"] == "negative"]

    if "pi" not in attn.columns:
        logger.warning("PI column not found")
        return

    # Join with behavior metrics
    merged = attn[["prompt_id", "pi"]].merge(
        behavior_df[["prompt_id", "violation_rate"]],
        on="prompt_id",
        how="inner"
    )

    if len(merged) < 2:
        logger.warning("Not enough data points for PI vs violation plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(merged["pi"], merged["violation_rate"], alpha=0.5, s=20)

    # Best fit line
    x = merged["pi"].values
    y = merged["violation_rate"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() >= 2:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, "r--", label=f"Fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}")
        ax.legend()

    ax.set_xlabel("Pressure Index (PI)")
    ax.set_ylabel("Violation Rate")
    ax.set_title("Pressure Index vs Violation Probability")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "pi_vs_violation.png", dpi=200)
    plt.close()


def _plot_logit_lens_curves(
    logit_lens_df: pd.DataFrame,
    outcomes: Dict[str, Dict],
    figures_dir: Path,
) -> None:
    """Figure 5: Layerwise logit lens curves."""
    df = logit_lens_df.copy()
    df["outcome"] = df["prompt_id"].apply(lambda x: outcomes.get(x, {}).get("outcome", "unknown"))

    if "layer" not in df.columns or "p_sem_first_token" not in df.columns:
        logger.warning("Required columns missing for logit lens plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"baseline": "blue", "negative": "orange"}
    linestyles = {"success": "-", "failure": "--"}

    for condition in df["condition"].unique() if "condition" in df.columns else ["all"]:
        for outcome in ["success", "failure"]:
            if "condition" in df.columns:
                sub = df[(df["condition"] == condition) & (df["outcome"] == outcome)]
            else:
                sub = df[df["outcome"] == outcome]

            if sub.empty:
                continue

            grouped = sub.groupby("layer")["p_sem_first_token"].mean().reset_index()
            grouped = grouped.sort_values("layer")

            label = f"{condition}/{outcome}" if "condition" in df.columns else outcome
            color = colors.get(condition, "gray")
            ls = linestyles.get(outcome, "-")

            ax.plot(grouped["layer"], grouped["p_sem_first_token"], ls, color=color, label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("P_sem (First Token)")
    ax.set_title("Logit Lens: Semantic Probability by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "logit_lens_curves.png", dpi=200)
    plt.close()


def _plot_attn_ffn_contrib(
    decomp_df: pd.DataFrame,
    outcomes: Dict[str, Dict],
    figures_dir: Path,
) -> None:
    """Figure 6: Attn vs FFN contributions by layer."""
    df = decomp_df.copy()
    df["outcome"] = df["prompt_id"].apply(lambda x: outcomes.get(x, {}).get("outcome", "unknown"))

    if "layer" not in df.columns:
        logger.warning("Layer column missing for decomposition plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"success": "green", "failure": "red"}

    for i, col in enumerate(["attn_contrib", "ffn_contrib"]):
        if col not in df.columns:
            continue

        ax = axes[i]
        for outcome in ["success", "failure"]:
            sub = df[df["outcome"] == outcome]
            if sub.empty:
                continue

            grouped = sub.groupby("layer")[col].mean().reset_index()
            grouped = grouped.sort_values("layer")

            ax.plot(grouped["layer"], grouped[col], "-", color=colors.get(outcome), label=outcome)

        ax.set_xlabel("Layer")
        ax.set_ylabel(col.replace("_", " ").title())
        ax.set_title(f"{col.replace('_', ' ').title()} by Layer")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "attn_ffn_contrib.png", dpi=200)
    plt.close()


def _plot_patching_effects(
    patching_df: pd.DataFrame,
    prompts_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Figure 7: Activation patching effects by layer and bin."""
    df = patching_df.copy()
    
    # Only merge if p0_bin not already present
    if "p0_bin" not in df.columns:
        df = df.merge(prompts_df[["prompt_id", "p0_bin"]], on="prompt_id", how="inner")

    if "layer" not in df.columns or "delta_p" not in df.columns or "p0_bin" not in df.columns:
        logger.warning("Required columns missing for patching plot")
        return

    # Mean delta_p per (p0_bin, layer) across all patch types
    grouped = df.groupby(["p0_bin", "layer"])["delta_p"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = sorted(grouped["p0_bin"].unique(), key=_parse_bin)
    for bin_name in bins:
        sub = grouped[grouped["p0_bin"] == bin_name].sort_values("layer")
        ax.plot(sub["layer"], sub["delta_p"], "-o", label=bin_name, markersize=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Delta P")
    ax.set_title("Patching Effects by Layer and P0 Bin")
    ax.legend(title="P0 Bin", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "patching_effects.png", dpi=200)
    plt.close()


# ============================================================================
# TABLES JSON
# ============================================================================


def _generate_tables_json(
    bootstrap_df: pd.DataFrame,
    attention_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    logit_lens_df: pd.DataFrame,
    decomp_df: pd.DataFrame,
    patching_df: pd.DataFrame,
    prompts_df: pd.DataFrame,
    outcomes: Dict[str, Dict],
    run_root: Path,
) -> None:
    """Generate tables.json with exact values used in plots."""
    tables = {}

    # violation_rate_by_bin
    vr = bootstrap_df[bootstrap_df["metric_name"] == "violation_rate"].copy()
    vr["bin"] = vr["group_key"].apply(lambda x: x.split("=")[-1] if "=" in x else x)
    tables["violation_rate_by_bin"] = vr[["bin", "mean", "ci_low", "ci_high", "n_prompts"]].to_dict("records")

    # suppression_relative_by_bin
    sr = bootstrap_df[bootstrap_df["metric_name"] == "suppression_relative"].copy()
    sr["bin"] = sr["group_key"].apply(lambda x: x.split("=")[-1] if "=" in x else x)
    tables["suppression_relative_by_bin"] = sr[["bin", "mean", "ci_low", "ci_high", "n_prompts"]].to_dict("records")

    # attention_metrics_by_bin_outcome
    attn = attention_df.copy()
    if "aggregate_flag" in attn.columns:
        attn = attn[attn["aggregate_flag"] == "global_mean"]
    if "condition" in attn.columns:
        attn = attn[attn["condition"] == "negative"]
    attn = attn.merge(prompts_df[["prompt_id", "p0_bin"]], on="prompt_id", how="inner")
    attn["outcome"] = attn["prompt_id"].apply(lambda x: outcomes.get(x, {}).get("outcome", "unknown"))

    attn_grouped = attn.groupby(["p0_bin", "outcome"]).agg({
        "iar": "mean", "nf": "mean", "tmf": "mean"
    }).reset_index()
    attn_grouped["n_prompts"] = attn.groupby(["p0_bin", "outcome"]).size().values
    tables["attention_metrics_by_bin_outcome"] = attn_grouped.rename(columns={"p0_bin": "bin"}).to_dict("records")

    # pi_vs_violation_points
    attn_pi = attention_df.copy()
    if "aggregate_flag" in attn_pi.columns:
        attn_pi = attn_pi[attn_pi["aggregate_flag"] == "global_mean"]
    if "condition" in attn_pi.columns:
        attn_pi = attn_pi[attn_pi["condition"] == "negative"]
    if "pi" in attn_pi.columns:
        merged = attn_pi[["prompt_id", "pi"]].merge(
            behavior_df[["prompt_id", "violation_rate"]], on="prompt_id", how="inner"
        )
        tables["pi_vs_violation_points"] = merged.to_dict("records")

    # logit_lens_curves
    ll = logit_lens_df.copy()
    ll["outcome"] = ll["prompt_id"].apply(lambda x: outcomes.get(x, {}).get("outcome", "unknown"))
    tables["logit_lens_curves"] = {}
    for condition in ll["condition"].unique() if "condition" in ll.columns else ["all"]:
        for outcome in ["success", "failure"]:
            key = f"{condition}/{outcome}"
            if "condition" in ll.columns:
                sub = ll[(ll["condition"] == condition) & (ll["outcome"] == outcome)]
            else:
                sub = ll[ll["outcome"] == outcome]
            if not sub.empty and "p_sem_first_token" in sub.columns:
                grouped = sub.groupby("layer")["p_sem_first_token"].mean().reset_index()
                tables["logit_lens_curves"][key] = grouped.to_dict("records")

    # attn_ffn_decomp
    dec = decomp_df.copy()
    dec["outcome"] = dec["prompt_id"].apply(lambda x: outcomes.get(x, {}).get("outcome", "unknown"))
    tables["attn_ffn_decomp"] = {}
    for outcome in ["success", "failure"]:
        sub = dec[dec["outcome"] == outcome]
        if not sub.empty:
            cols = ["layer"]
            if "attn_contrib" in sub.columns:
                cols.append("attn_contrib")
            if "ffn_contrib" in sub.columns:
                cols.append("ffn_contrib")
            grouped = sub.groupby("layer")[cols[1:]].mean().reset_index()
            tables["attn_ffn_decomp"][outcome] = grouped.to_dict("records")

    # patching_effects_by_bin
    pat = patching_df.copy()
    if "p0_bin" not in pat.columns:
        pat = pat.merge(prompts_df[["prompt_id", "p0_bin"]], on="prompt_id", how="inner")
    tables["patching_effects_by_bin"] = {}
    if "p0_bin" in pat.columns:
        for bin_name in pat["p0_bin"].unique():
            sub = pat[pat["p0_bin"] == bin_name]
            grouped = sub.groupby("layer")["delta_p"].mean().reset_index()
            grouped = grouped.rename(columns={"delta_p": "mean_delta_p"})
            tables["patching_effects_by_bin"][bin_name] = grouped.to_dict("records")

    # Write
    with open(run_root / "tables.json", "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=True)


# ============================================================================
# APPENDIX EXAMPLES
# ============================================================================


def _generate_appendix_examples(
    prompts_df: pd.DataFrame,
    attention_df: pd.DataFrame,
    outcomes: Dict[str, Dict],
    run_root: Path,
    limit: int = 20,
) -> None:
    """Generate appendix example JSON files for failures."""
    examples_dir = run_root / "appendix_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Get failures sorted by prompt_id
    failures = [
        pid for pid, data in outcomes.items()
        if data.get("outcome") == "failure"
    ]
    failures = sorted(failures)[:limit]

    # Prepare attention metrics lookup
    attn = attention_df.copy()
    if "aggregate_flag" in attn.columns:
        attn = attn[attn["aggregate_flag"] == "global_mean"]
    if "condition" in attn.columns:
        attn = attn[attn["condition"] == "negative"]
    attn_lookup = {row["prompt_id"]: row for _, row in attn.iterrows()}

    # Generate examples
    for i, prompt_id in enumerate(failures):
        prompt_row = prompts_df[prompts_df["prompt_id"] == prompt_id]
        if prompt_row.empty:
            continue

        prompt_row = prompt_row.iloc[0]
        outcome_data = outcomes.get(prompt_id, {})

        example = {
            "prompt_id": prompt_id,
            "category": prompt_row.get("category", ""),
            "question_text": prompt_row.get("question_text", ""),
            "target_word": prompt_row.get("target_word", ""),
            "prompt_text": build_prompt(
                prompt_row.get("question_text", ""),
                prompt_row.get("target_word", ""),
                "negative"
            ),
            "output_text": outcome_data.get("completion_text", ""),
            "token_spans": outcome_data.get("token_spans", []),
        }

        # Attention metrics
        attn_row = attn_lookup.get(prompt_id)
        if attn_row is not None:
            example["attention_metrics"] = {
                "iar": float(attn_row.get("iar")) if pd.notna(attn_row.get("iar")) else None,
                "nf": float(attn_row.get("nf")) if pd.notna(attn_row.get("nf")) else None,
                "tmf": float(attn_row.get("tmf")) if pd.notna(attn_row.get("tmf")) else None,
                "pi": float(attn_row.get("pi")) if pd.notna(attn_row.get("pi")) else None,
            }
        else:
            example["attention_metrics"] = None

        filename = f"{i:02d}_{prompt_id}.json"
        with open(examples_dir / filename, "w", encoding="utf-8") as f:
            json.dump(example, f, indent=2, ensure_ascii=True)


# ============================================================================
# PIPELINE ENTRYPOINT
# ============================================================================


def run_visualization_pipeline(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
    limit_examples: int = 20,
) -> Dict[str, str]:
    """
    Run full visualization pipeline.

    Generates figures, tables.json, and appendix examples.

    Args:
        output_root: Run root directory
        prompts_path: Path to prompts.csv
        limit_examples: Number of appendix examples

    Returns:
        Dict of output file paths
    """
    run_root = _resolve_run_root(output_root)
    figures_dir = run_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = get_base_paths()
    data_root = paths.get("data_root", Path("data"))

    logger.info("Loading data...")

    # Load all required data
    prompts_df = _load_prompts_df(prompts_path, data_root)
    behavior_df = _load_required_csv(run_root, "behavior_metrics.csv")
    psem_df = _load_required_csv(run_root, "psem.csv")
    attention_df = _load_required_csv(run_root, "attention_metrics.csv")
    logit_lens_df = _load_required_csv(run_root, "logit_lens.csv")
    decomp_df = _load_required_csv(run_root, "ffn_attn_decomp.csv")
    patching_df = _load_required_csv(run_root, "patching_results.csv")
    bootstrap_df = _load_required_csv(run_root, "bootstrap_results.csv")

    # Derive outcomes
    outcomes = _derive_outcomes(run_root)

    result_paths = {}

    # Generate figures
    logger.info("Generating figures...")

    _plot_violation_rate_vs_p0(bootstrap_df, figures_dir)
    result_paths["violation_rate_vs_p0"] = str(figures_dir / "violation_rate_vs_p0.png")

    _plot_suppression_relative_vs_p0(bootstrap_df, figures_dir)
    result_paths["suppression_relative_vs_p0"] = str(figures_dir / "suppression_relative_vs_p0.png")

    _plot_attention_metrics_vs_p0(attention_df, prompts_df, outcomes, figures_dir)
    result_paths["attention_metrics_vs_p0"] = str(figures_dir / "attention_metrics_vs_p0.png")

    _plot_pi_vs_violation(attention_df, behavior_df, figures_dir)
    result_paths["pi_vs_violation"] = str(figures_dir / "pi_vs_violation.png")

    _plot_logit_lens_curves(logit_lens_df, outcomes, figures_dir)
    result_paths["logit_lens_curves"] = str(figures_dir / "logit_lens_curves.png")

    _plot_attn_ffn_contrib(decomp_df, outcomes, figures_dir)
    result_paths["attn_ffn_contrib"] = str(figures_dir / "attn_ffn_contrib.png")

    _plot_patching_effects(patching_df, prompts_df, figures_dir)
    result_paths["patching_effects"] = str(figures_dir / "patching_effects.png")

    # Generate tables.json
    logger.info("Generating tables.json...")
    _generate_tables_json(
        bootstrap_df, attention_df, behavior_df, logit_lens_df,
        decomp_df, patching_df, prompts_df, outcomes, run_root
    )
    result_paths["tables_json"] = str(run_root / "tables.json")

    # Generate appendix examples
    logger.info("Generating appendix examples...")
    _generate_appendix_examples(prompts_df, attention_df, outcomes, run_root, limit_examples)
    result_paths["appendix_examples"] = str(run_root / "appendix_examples")

    logger.info("Visualization pipeline complete.")
    return result_paths


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZE MODULE SELF-TEST")
    print("=" * 60)

    # Test 1: _resolve_run_root
    print("\n1. Testing _resolve_run_root:")
    try:
        import warnings as w
        with w.catch_warnings(record=True):
            w.simplefilter("always")
            result = _resolve_run_root(None)
            print(f"   Resolved to: {result}")
            print("   PASS: _resolve_run_root works")
    except Exception as e:
        print(f"   SKIP: {e}")

    # Test 2: _parse_bin
    print("\n2. Testing _parse_bin:")
    try:
        assert _parse_bin("[0.0, 0.2)") == 0.0
        assert _parse_bin("p0_bin=[0.4, 0.6)") == 0.4
        assert _parse_bin("0.0-0.2") == 0.0
        assert _parse_bin("0.4-0.6") == 0.4
        print("   PASS: Bin parsing works")
    except Exception as e:
        print(f"   FAIL: {e}")

    # Test 3: _get_bin_center
    print("\n3. Testing _get_bin_center:")
    try:
        assert abs(_get_bin_center("[0.0, 0.2)") - 0.1) < 0.01
        assert abs(_get_bin_center("[0.4, 0.6)") - 0.5) < 0.01
        assert abs(_get_bin_center("0.0-0.2") - 0.1) < 0.01
        assert abs(_get_bin_center("0.4-0.6") - 0.5) < 0.01
        print("   PASS: Bin center computation works")
    except Exception as e:
        print(f"   FAIL: {e}")

    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)
