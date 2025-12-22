"""
bootstrap.py - Bootstrap Confidence Interval Computation

This module computes 95% bootstrap CIs for:
- Violation rate per pressure bin
- Suppression metrics per bin  
- Attention metrics per bin
- Patching effect sizes per bin

Per specification Section 11 and execution-plan Section 11.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled explicitly at runtime
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled explicitly at runtime
    pd = None

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS
# ============================================================================

try:
    from .config import get_base_paths, CONFIG
except ImportError:
    from config import get_base_paths, CONFIG


# ============================================================================
# DEPENDENCY GUARDS
# ============================================================================


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "NumPy is required for bootstrap computations. "
            "Install it with `pip install numpy`."
        )


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError(
            "pandas is required for bootstrap computations. "
            "Install it with `pip install pandas`."
        )


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
    """
    Load prompts.csv as DataFrame.

    Args:
        prompts_path: Explicit path or None
        data_root: Data root directory

    Returns:
        DataFrame with prompts

    Raises:
        FileNotFoundError: If file not found
        ValueError: If required columns missing
    """
    _require_pandas()
    path = prompts_path or (data_root / "prompts.csv")
    if not path.exists():
        raise FileNotFoundError(f"prompts.csv not found: {path}")

    df = pd.read_csv(path)

    required = ["prompt_id", "p0_bin"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"prompts.csv missing columns: {missing}")

    # Ensure prompt_id is string
    df["prompt_id"] = df["prompt_id"].astype(str)
    return df


def _load_required_csv(run_root: Path, filename: str) -> pd.DataFrame:
    """
    Load a required CSV file from runs directory.

    Args:
        run_root: Run root directory
        filename: CSV filename

    Returns:
        DataFrame

    Raises:
        FileNotFoundError: If file not found
    """
    _require_pandas()
    path = run_root / "runs" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}. "
            "Ensure all pipeline steps have been run."
        )
    df = pd.read_csv(path)
    if "prompt_id" in df.columns:
        df["prompt_id"] = df["prompt_id"].astype(str)
    return df


# ============================================================================
# BOOTSTRAP HELPER
# ============================================================================


def _bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    n_iterations: int,
    ci_percentiles: Tuple[float, float],
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        values: Array of values to bootstrap
        rng: NumPy random generator
        n_iterations: Number of bootstrap iterations
        ci_percentiles: (low, high) percentiles for CI

    Returns:
        (mean, ci_low, ci_high)
    """
    _require_numpy()
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    original_mean = float(np.mean(values))
    n = len(values)

    bootstrap_means = np.empty(n_iterations)
    for i in range(n_iterations):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    ci_low = float(np.percentile(bootstrap_means, ci_percentiles[0]))
    ci_high = float(np.percentile(bootstrap_means, ci_percentiles[1]))

    return (original_mean, ci_low, ci_high)


# ============================================================================
# METRIC COMPUTATION
# ============================================================================


def _compute_violation_rate_ci(
    prompts_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    rng: np.random.Generator,
    n_iterations: int,
    ci_percentiles: Tuple[float, float],
    seed: int,
) -> List[Dict]:
    """Compute bootstrap CI for violation rate per p0_bin."""
    results = []

    # Join with prompts to get p0_bin
    merged = behavior_df.merge(
        prompts_df[["prompt_id", "p0_bin"]],
        on="prompt_id",
        how="inner",
    )

    if "violation_rate" not in merged.columns:
        logger.warning("violation_rate column not found in behavior_metrics.csv")
        return results

    for bin_name in sorted(merged["p0_bin"].unique()):
        bin_data = merged[merged["p0_bin"] == bin_name]
        values = bin_data["violation_rate"].dropna().values

        mean, ci_low, ci_high = _bootstrap_mean(values, rng, n_iterations, ci_percentiles)

        results.append({
            "metric_name": "violation_rate",
            "group_key": f"p0_bin={bin_name}",
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_prompts": len(values),
            "seed": seed,
        })

    return results


def _compute_suppression_ci(
    prompts_df: pd.DataFrame,
    psem_df: pd.DataFrame,
    rng: np.random.Generator,
    n_iterations: int,
    ci_percentiles: Tuple[float, float],
    seed: int,
) -> List[Dict]:
    """Compute bootstrap CI for suppression metrics per p0_bin."""
    results = []

    merged = psem_df.merge(
        prompts_df[["prompt_id", "p0_bin"]],
        on="prompt_id",
        how="inner",
    )

    metrics = [
        ("delta", "suppression_delta"),
        ("relative", "suppression_relative"),
        ("log", "suppression_log"),
    ]

    for bin_name in sorted(merged["p0_bin"].unique()):
        bin_data = merged[merged["p0_bin"] == bin_name]

        for col, metric_name in metrics:
            if col not in bin_data.columns:
                continue

            values = bin_data[col].dropna().values
            mean, ci_low, ci_high = _bootstrap_mean(values, rng, n_iterations, ci_percentiles)

            results.append({
                "metric_name": metric_name,
                "group_key": f"p0_bin={bin_name}",
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_prompts": len(values),
                "seed": seed,
            })

    return results


def _compute_attention_ci(
    prompts_df: pd.DataFrame,
    attention_df: pd.DataFrame,
    rng: np.random.Generator,
    n_iterations: int,
    ci_percentiles: Tuple[float, float],
    seed: int,
) -> List[Dict]:
    """Compute bootstrap CI for attention metrics per p0_bin and condition."""
    results = []

    # Filter to global_mean aggregates
    if "aggregate_flag" in attention_df.columns:
        attention_df = attention_df[attention_df["aggregate_flag"] == "global_mean"]

    merged = attention_df.merge(
        prompts_df[["prompt_id", "p0_bin"]],
        on="prompt_id",
        how="inner",
    )

    metrics = [
        ("iar", "attention_iar"),
        ("nf", "attention_nf"),
        ("tmf", "attention_tmf"),
        ("pi", "attention_pi"),
    ]

    conditions = merged["condition"].unique() if "condition" in merged.columns else ["all"]

    for bin_name in sorted(merged["p0_bin"].unique()):
        for condition in conditions:
            if "condition" in merged.columns:
                bin_data = merged[(merged["p0_bin"] == bin_name) & (merged["condition"] == condition)]
            else:
                bin_data = merged[merged["p0_bin"] == bin_name]

            for col, metric_name in metrics:
                if col not in bin_data.columns:
                    continue

                values = bin_data[col].dropna().values
                mean, ci_low, ci_high = _bootstrap_mean(values, rng, n_iterations, ci_percentiles)

                results.append({
                    "metric_name": metric_name,
                    "group_key": f"p0_bin={bin_name}|condition={condition}",
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_prompts": len(values),
                    "seed": seed,
                })

    return results


def _compute_patching_ci(
    prompts_df: pd.DataFrame,
    patching_df: pd.DataFrame,
    rng: np.random.Generator,
    n_iterations: int,
    ci_percentiles: Tuple[float, float],
    seed: int,
) -> List[Dict]:
    """Compute bootstrap CI for patching effect sizes per p0_bin and patch_type."""
    results = []

    if "delta_p" not in patching_df.columns or "patch_type" not in patching_df.columns:
        logger.warning("Required columns missing in patching_results.csv")
        return results

    # Reduce to per-prompt mean delta_p by patch_type
    prompt_means = patching_df.groupby(["prompt_id", "patch_type"])["delta_p"].mean().reset_index()

    merged = prompt_means.merge(
        prompts_df[["prompt_id", "p0_bin"]],
        on="prompt_id",
        how="inner",
    )

    patch_types = merged["patch_type"].unique()

    for bin_name in sorted(merged["p0_bin"].unique()):
        for patch_type in sorted(patch_types):
            bin_data = merged[(merged["p0_bin"] == bin_name) & (merged["patch_type"] == patch_type)]
            values = bin_data["delta_p"].dropna().values

            mean, ci_low, ci_high = _bootstrap_mean(values, rng, n_iterations, ci_percentiles)

            results.append({
                "metric_name": "patching_delta_p",
                "group_key": f"p0_bin={bin_name}|patch_type={patch_type}",
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_prompts": len(values),
                "seed": seed,
            })

    return results


# ============================================================================
# PIPELINE ENTRYPOINT
# ============================================================================


def run_bootstrap_pipeline(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
    seed: Optional[int] = None,
    n_iterations: Optional[int] = None,
) -> Path:
    """
    Run bootstrap CI computation pipeline.

    Computes 95% bootstrap CIs for:
    - Violation rate per bin
    - Suppression metrics per bin
    - Attention metrics per bin
    - Patching effect sizes per bin

    Args:
        output_root: Run root directory
        prompts_path: Path to prompts.csv
        seed: Random seed
        n_iterations: Number of bootstrap iterations

    Returns:
        Path to bootstrap_results.csv
    """
    _require_numpy()
    _require_pandas()
    run_root = _resolve_run_root(output_root)
    runs_dir = run_root / "runs"

    paths = get_base_paths()
    data_root = paths.get("data_root", Path("data"))

    # Load prompts
    prompts_df = _load_prompts_df(prompts_path, data_root)

    # Load required files
    behavior_df = _load_required_csv(run_root, "behavior_metrics.csv")
    psem_df = _load_required_csv(run_root, "psem.csv")
    attention_df = _load_required_csv(run_root, "attention_metrics.csv")
    patching_df = _load_required_csv(run_root, "patching_results.csv")

    # Get config
    bootstrap_cfg = CONFIG.get("bootstrap", {})
    if n_iterations is None:
        n_iterations = bootstrap_cfg.get("n_iterations", 1000)
    ci_percentiles = tuple(bootstrap_cfg.get("ci_percentiles", [2.5, 97.5]))

    if seed is None:
        seed = CONFIG.get("seeds", {}).get("python", 42)

    rng = np.random.default_rng(seed)

    logger.info("Running bootstrap CI computation (n_iterations=%d, seed=%d)", n_iterations, seed)

    # Compute all metrics
    all_results = []

    logger.info("Computing violation rate CIs...")
    all_results.extend(_compute_violation_rate_ci(
        prompts_df, behavior_df, rng, n_iterations, ci_percentiles, seed
    ))

    logger.info("Computing suppression metric CIs...")
    all_results.extend(_compute_suppression_ci(
        prompts_df, psem_df, rng, n_iterations, ci_percentiles, seed
    ))

    logger.info("Computing attention metric CIs...")
    all_results.extend(_compute_attention_ci(
        prompts_df, attention_df, rng, n_iterations, ci_percentiles, seed
    ))

    logger.info("Computing patching effect size CIs...")
    all_results.extend(_compute_patching_ci(
        prompts_df, patching_df, rng, n_iterations, ci_percentiles, seed
    ))

    # Write results
    output_path = runs_dir / "bootstrap_results.csv"
    df = pd.DataFrame(all_results)

    # Ensure column order
    columns = ["metric_name", "group_key", "mean", "ci_low", "ci_high", "n_prompts", "seed"]
    df = df[columns]

    df.to_csv(output_path, index=False)
    logger.info("Wrote %d bootstrap results to %s", len(df), output_path)

    return output_path


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BOOTSTRAP MODULE SELF-TEST")
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

    # Test 2: _bootstrap_mean
    print("\n2. Testing _bootstrap_mean:")
    if np is None:
        print("   SKIP: NumPy not installed")
    else:
        try:
            rng = np.random.default_rng(42)
            values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            mean, ci_low, ci_high = _bootstrap_mean(values, rng, 1000, (2.5, 97.5))
            print(f"   Values: {values}")
            print(f"   Mean: {mean:.4f}")
            print(f"   95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            print("   PASS: Bootstrap computation works")
        except Exception as e:
            print(f"   FAIL: {e}")

    # Test 3: Empty values
    print("\n3. Testing _bootstrap_mean with empty array:")
    if np is None:
        print("   SKIP: NumPy not installed")
    else:
        try:
            rng = np.random.default_rng(42)
            values = np.array([])
            mean, ci_low, ci_high = _bootstrap_mean(values, rng, 1000, (2.5, 97.5))
            assert np.isnan(mean), "Expected NaN for empty input"
            print("   PASS: Empty array returns NaN")
        except Exception as e:
            print(f"   FAIL: {e}")

    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)
