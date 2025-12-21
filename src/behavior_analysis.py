"""
behavior_analysis.py - Detection, Behavioral Metrics, and PSEM/Bins Pipeline

This module provides:
- Detection and token-span mapping for completions
- Behavioral metrics (violation_rate, format_adherence, clean_success_rate)
- PSEM and pressure bins outputs

Per specification Sections 5, 8, and execution-plan Sections 8-9.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Generator

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS
# ============================================================================

try:
    from .config import CONFIG, get_base_paths
    from .utils import ModelWrapper, normalize_for_match
    from .detector import detect_and_map
    from .metrics_psem import compute_suppression_metrics, token_sequences_for_variants
except ImportError:
    from config import CONFIG, get_base_paths
    from utils import ModelWrapper, normalize_for_match
    from detector import detect_and_map
    from metrics_psem import compute_suppression_metrics, token_sequences_for_variants


# ============================================================================
# UTILITIES
# ============================================================================

def load_jsonl(path: Path) -> Generator[Dict, None, None]:
    """
    Load JSONL file as generator of dicts.

    Args:
        path: Path to JSONL file

    Yields:
        Dict for each line
    """
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON line in %s: %s", path, e)


def load_prompts_df(path: Path) -> "pd.DataFrame":
    """
    Load prompts.csv as pandas DataFrame.

    Args:
        path: Path to prompts.csv

    Returns:
        pandas DataFrame with prompts

    Raises:
        ValueError: If required columns are missing
    """
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"prompts.csv not found: {path}")

    df = pd.read_csv(path)

    required_cols = ["prompt_id", "question_text", "target_word", "category", "p0", "p1"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"prompts.csv missing required columns: {missing}")

    return df


def _append_jsonl(record: Dict, path: Path) -> None:
    """Append a single record to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # Do not fail on logging errors


def _resolve_run_root(output_root: Optional[Path]) -> Path:
    """
    Resolve the run root directory from output_root or find the latest run.

    Priority order:
    1) If output_root name starts with experiment_run_, return it (specific run).
    2) If output_root contains experiment_run_* subdirs, select latest by name.
    3) If output_root has runs/ subdir (but no experiment_run_* children), use it.
    4) Otherwise, return output_root as-is (backward compatibility).
    5) If output_root is None, select latest experiment_run_* under base outputs.

    Args:
        output_root: Optional run root or base outputs directory

    Returns:
        Resolved run root Path
    """
    paths = get_base_paths()
    base_out = paths.get("output_root", Path("outputs"))

    def find_latest_run_in(parent: Path) -> Optional[Path]:
        """Find latest experiment_run_* subdir by lexicographic name."""
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

        # Priority 1: If name starts with experiment_run_, it IS a run root
        if candidate.name.startswith("experiment_run_"):
            return candidate

        # Priority 2: Check for experiment_run_* subdirs BEFORE checking runs/
        latest = find_latest_run_in(candidate)
        if latest:
            logger.warning(
                "Selecting latest experiment_run_* under %s: %s",
                candidate, latest,
            )
            return latest

        # Priority 3: If it has runs/ but no experiment_run_*, treat as run root
        if candidate.exists() and (candidate / "runs").exists():
            return candidate

        # Priority 4: Backward compatibility - return as-is
        return candidate

    # output_root is None: find latest run under base_out
    latest = find_latest_run_in(base_out)
    if latest:
        logger.warning("output_root not provided; using latest run dir: %s", latest)
        return latest

    # Fallback to base_out
    return base_out


# ============================================================================
# DETECTION + MAPPING PIPELINE
# ============================================================================

def run_detection_mapping(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run detection and token-span mapping on completions.

    Per specification Section 5 and execution-plan Section 8:
    - Loads prompts.csv and completions JSONL files
    - Runs detect_and_map for each completion
    - Writes detection_mapping.jsonl
    - Hard halts if mapping errors exceed threshold

    Args:
        output_root: Run root directory (e.g., outputs/experiment_run_YYYYMMDD_HHMM).
                     If None, the latest experiment_run_* under base outputs is used.
                     If pointing to base outputs with experiment_run_* children,
                     the latest run is auto-selected.
                     Completions and outputs are read/written under run_root/runs.
        prompts_path: Path to prompts.csv. If None, uses data_root/prompts.csv
                      from get_base_paths().

    Returns:
        Dict with counts and status

    Raises:
        FileNotFoundError: If prompts.csv not found
        RuntimeError: If mapping errors exceed threshold (>0 or >0.1%)
    """
    paths = get_base_paths()
    # data_root is ALWAYS from get_base_paths, never from output_root
    data_root = paths.get("data_root", Path("data"))

    # Resolve run root (handles None or base outputs automatically)
    run_root = _resolve_run_root(output_root)
    runs_dir = run_root / "runs"
    errors_dir = run_root / "errors"

    runs_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts.csv
    prompts_path = prompts_path or (data_root / "prompts.csv")
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"prompts.csv not found at {prompts_path}; "
            "pass prompts_path or place it under data_root"
        )
    prompts_df = load_prompts_df(prompts_path)
    prompt_targets = {
        row["prompt_id"]: row["target_word"]
        for _, row in prompts_df.iterrows()
    }

    # Load tokenizer (model not required for mapping)
    try:
        wrapper = ModelWrapper.get_instance()
        if wrapper.tokenizer is None:
            wrapper.load()
        tokenizer = wrapper.tokenizer
    except Exception as e:
        logger.error("Failed to load tokenizer: %s", e)
        raise RuntimeError(f"Tokenizer required for detection mapping: {e}")

    # Check for existing detection_mapping.jsonl and build processed set
    detection_path = runs_dir / "detection_mapping.jsonl"
    processed_keys: Set[str] = set()
    if detection_path.exists():
        for record in load_jsonl(detection_path):
            key = f"{record.get('prompt_id')}|{record.get('condition')}|{record.get('sample_id', '')}"
            processed_keys.add(key)
        logger.info("Resuming: %d records already processed", len(processed_keys))

    # Files to process
    greedy_path = runs_dir / "completions_greedy.jsonl"
    samples_path = runs_dir / "completions_samples.jsonl"
    errors_path = errors_dir / "mapping_errors.jsonl"

    mapping_error_count = 0
    total_count = 0

    def process_completion(record: Dict, sample_id: str = "") -> None:
        nonlocal mapping_error_count, total_count

        prompt_id = record.get("prompt_id")
        condition = record.get("condition")
        generated_text = record.get("generated_text", "")
        generated_token_ids = record.get("generated_token_ids", [])

        # Build key for resumability
        key = f"{prompt_id}|{condition}|{sample_id}"
        if key in processed_keys:
            return

        total_count += 1

        # Get target word
        target_word = prompt_targets.get(prompt_id, "")
        if not target_word:
            logger.warning("No target word for prompt_id=%s", prompt_id)
            return

        # Handle missing/empty token IDs
        if not generated_token_ids:
            # mapping error - log and continue
            mapping_error_count += 1
            error_record = {
                "prompt_id": prompt_id,
                "condition": condition,
                "sample_id": sample_id,
                "error_reason": "missing_or_empty_token_ids",
                "generated_text": generated_text[:500] if generated_text else "",
            }
            _append_jsonl(error_record, errors_path)

            # Still write detection record with mapping_error=True
            result = {
                "prompt_id": prompt_id,
                "condition": condition,
                "sample_id": sample_id,
                "completion_text": generated_text,
                "completion_norm": normalize_for_match(generated_text),
                "target_word": target_word,
                "target_word_norm": normalize_for_match(target_word),
                "word_present": False,
                "token_spans": [],
                "mapping_error": True,
                "format_adherence": False,
            }
            _append_jsonl(result, detection_path)
            processed_keys.add(key)
            return

        # Run detection and mapping
        detection = detect_and_map(
            target=target_word,
            completion_text=generated_text,
            token_ids=generated_token_ids,
            tokenizer=tokenizer,
            errors_path=str(errors_path),
            prompt_id=prompt_id,
            condition=condition,
        )

        if detection.get("mapping_error"):
            mapping_error_count += 1

        # Compute normalized strings
        completion_norm = normalize_for_match(generated_text)
        target_word_norm = normalize_for_match(target_word)

        # Format adherence: one word after normalization
        words = completion_norm.split()
        format_adherence = len(words) == 1

        # Write record
        result = {
            "prompt_id": prompt_id,
            "condition": condition,
            "sample_id": sample_id,
            "completion_text": generated_text,
            "completion_norm": completion_norm,
            "target_word": target_word,
            "target_word_norm": target_word_norm,
            "word_present": detection.get("word_present", False),
            "token_spans": detection.get("token_spans", []),
            "mapping_error": detection.get("mapping_error", False),
            "format_adherence": format_adherence,
        }
        _append_jsonl(result, detection_path)
        processed_keys.add(key)

    # Process greedy completions
    if greedy_path.exists():
        logger.info("Processing greedy completions from %s", greedy_path)
        for record in load_jsonl(greedy_path):
            process_completion(record, sample_id="")
    else:
        logger.warning("Greedy completions not found: %s", greedy_path)

    # Process sampled completions
    if samples_path.exists():
        logger.info("Processing sampled completions from %s", samples_path)
        for record in load_jsonl(samples_path):
            sample_id = str(record.get("sample_id", ""))
            process_completion(record, sample_id=sample_id)
    else:
        logger.warning("Sampled completions not found: %s", samples_path)

    # Hard halt check
    if total_count > 0:
        error_rate = mapping_error_count / total_count
        if mapping_error_count > 0 or error_rate > 0.001:
            raise RuntimeError(
                f"Mapping errors exceed threshold: {mapping_error_count} errors "
                f"({error_rate:.2%} rate). Check {errors_path}"
            )

    logger.info("Detection mapping complete: %d processed, %d errors", total_count, mapping_error_count)
    return {
        "total_processed": total_count,
        "mapping_errors": mapping_error_count,
        "detection_mapping_path": str(detection_path),
    }


# ============================================================================
# BEHAVIORAL METRICS
# ============================================================================

def compute_behavioral_metrics(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
) -> "pd.DataFrame":
    """
    Compute behavioral metrics from detection_mapping.jsonl.

    Per specification Section 8.2:
    - violation_rate: fraction with word_present == True
    - format_adherence_rate: fraction with format_adherence == True
    - clean_success_rate: fraction (not word_present AND format_adherence)

    Uses only negative condition samples (sample_id != "").

    Args:
        output_root: Run root directory (e.g., outputs/experiment_run_YYYYMMDD_HHMM).
                     If None, the latest experiment_run_* under base outputs is used.
                     If pointing to base outputs with experiment_run_* children,
                     the latest run is auto-selected.
                     detection_mapping.jsonl is read from run_root/runs.
        prompts_path: Path to prompts.csv. If None, uses data_root/prompts.csv
                      from get_base_paths().

    Returns:
        DataFrame with behavior_metrics
    """
    import pandas as pd
    from collections import defaultdict

    paths = get_base_paths()
    run_root = _resolve_run_root(output_root)
    runs_dir = run_root / "runs"

    detection_path = runs_dir / "detection_mapping.jsonl"
    if not detection_path.exists():
        raise FileNotFoundError(f"detection_mapping.jsonl not found: {detection_path}")

    # Load prompts for category lookup - use prompts_path if provided, else data_root
    data_root = paths.get("data_root", Path("data"))
    prompts_path = prompts_path or (data_root / "prompts.csv")
    prompts_df = load_prompts_df(prompts_path)
    prompt_categories = dict(zip(prompts_df["prompt_id"], prompts_df["category"]))

    # Group samples by prompt_id for negative condition only
    prompt_samples: Dict[str, List[Dict]] = defaultdict(list)
    for record in load_jsonl(detection_path):
        # Only negative condition samples (not greedy)
        if record.get("condition") != "negative":
            continue
        if record.get("sample_id", "") == "":
            continue

        prompt_id = record.get("prompt_id")
        prompt_samples[prompt_id].append(record)

    # Compute metrics per prompt
    rows = []
    for prompt_id, samples in prompt_samples.items():
        if not samples:
            continue

        n = len(samples)
        violations = sum(1 for s in samples if s.get("word_present", False))
        format_ok = sum(1 for s in samples if s.get("format_adherence", False))
        clean = sum(
            1 for s in samples
            if not s.get("word_present", False) and s.get("format_adherence", False)
        )

        rows.append({
            "prompt_id": prompt_id,
            "category": prompt_categories.get(prompt_id, ""),
            "violation_rate": violations / n,
            "format_adherence_rate": format_ok / n,
            "clean_success_rate": clean / n,
        })

    df = pd.DataFrame(rows)

    # Write to CSV
    metrics_path = runs_dir / "behavior_metrics.csv"
    df.to_csv(metrics_path, index=False)
    logger.info("Wrote %d behavior metrics to %s", len(df), metrics_path)

    return df


# ============================================================================
# PSEM + PRESSURE BINS OUTPUTS
# ============================================================================

def write_psem_and_bins(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Write psem.csv and pressure_bins.csv from prompts.csv.

    Per specification Sections 7, 8 and execution-plan Sections 6, 13:
    - psem.csv: prompt_id, category, p0, p1, delta, relative, log, num_token_sequences
    - pressure_bins.csv: prompt_id, category, p0, bin_id, bin_lower, bin_upper, p0_bin

    Args:
        output_root: Run root directory (e.g., outputs/experiment_run_YYYYMMDD_HHMM).
                     If None, the latest experiment_run_* under base outputs is used.
                     If pointing to base outputs with experiment_run_* children,
                     the latest run is auto-selected.
                     Output CSVs are written to run_root/runs.
        prompts_path: Path to prompts.csv. If None, uses data_root/prompts.csv
                      from get_base_paths().

    Returns:
        Dict with paths to output files
    """
    import pandas as pd

    paths = get_base_paths()
    # data_root is ALWAYS from get_base_paths
    data_root = paths.get("data_root", Path("data"))
    run_root = _resolve_run_root(output_root)
    runs_dir = run_root / "runs"

    runs_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts - use prompts_path if provided, else data_root
    prompts_path = prompts_path or (data_root / "prompts.csv")
    prompts_df = load_prompts_df(prompts_path)

    # Load tokenizer for token sequences (if available)
    tokenizer = None
    try:
        wrapper = ModelWrapper.get_instance()
        if wrapper.tokenizer is None:
            try:
                wrapper.load()
            except Exception:
                pass
        tokenizer = wrapper.tokenizer
    except Exception as e:
        logger.warning("Tokenizer unavailable for token sequence count: %s", e)

    # Cache token sequence counts by target word
    token_seq_cache: Dict[str, int] = {}

    def get_num_token_sequences(target_word: str) -> int:
        if target_word in token_seq_cache:
            return token_seq_cache[target_word]

        if tokenizer is None:
            token_seq_cache[target_word] = 0
            return 0

        try:
            seqs = token_sequences_for_variants(target_word, tokenizer)
            count = len(seqs)
        except Exception:
            count = 0

        token_seq_cache[target_word] = count
        return count

    # Get pressure bins from config
    pressure_bins = CONFIG['dataset'].get('pressure_bins', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def compute_bin_info(p0: float) -> Dict[str, Any]:
        """Compute bin info for p0 value."""
        p0_clamped = max(0.0, min(1.0, p0))
        for i in range(len(pressure_bins) - 1):
            lower = pressure_bins[i]
            upper = pressure_bins[i + 1]
            # Last bin includes 1.0
            if i == len(pressure_bins) - 2:
                if lower <= p0_clamped <= upper:
                    return {
                        "bin_id": i,
                        "bin_lower": lower,
                        "bin_upper": upper,
                        "p0_bin": f"{lower:.1f}-{upper:.1f}",
                    }
            else:
                if lower <= p0_clamped < upper:
                    return {
                        "bin_id": i,
                        "bin_lower": lower,
                        "bin_upper": upper,
                        "p0_bin": f"{lower:.1f}-{upper:.1f}",
                    }
        # Fallback: last bin
        return {
            "bin_id": len(pressure_bins) - 2,
            "bin_lower": pressure_bins[-2],
            "bin_upper": pressure_bins[-1],
            "p0_bin": f"{pressure_bins[-2]:.1f}-{pressure_bins[-1]:.1f}",
        }

    # Build psem.csv rows
    psem_rows = []
    bins_rows = []

    for _, row in prompts_df.iterrows():
        prompt_id = row["prompt_id"]
        category = row["category"]
        target_word = row["target_word"]
        p0 = float(row["p0"])
        p1 = float(row["p1"])

        # Compute suppression metrics
        supp = compute_suppression_metrics(p0, p1)

        # Get token sequence count
        num_seqs = get_num_token_sequences(target_word)

        psem_rows.append({
            "prompt_id": prompt_id,
            "category": category,
            "p0": p0,
            "p1": p1,
            "delta": supp["delta"],
            "relative": supp["relative"],
            "log": supp["log_suppression"],
            "num_token_sequences": num_seqs,
        })

        # Bin info
        bin_info = compute_bin_info(p0)
        bins_rows.append({
            "prompt_id": prompt_id,
            "category": category,
            "p0": p0,
            "bin_id": bin_info["bin_id"],
            "bin_lower": bin_info["bin_lower"],
            "bin_upper": bin_info["bin_upper"],
            "p0_bin": bin_info["p0_bin"],
        })

    # Write CSVs
    psem_df = pd.DataFrame(psem_rows)
    psem_path = runs_dir / "psem.csv"
    psem_df.to_csv(psem_path, index=False)
    logger.info("Wrote %d rows to %s", len(psem_df), psem_path)

    bins_df = pd.DataFrame(bins_rows)
    bins_path = runs_dir / "pressure_bins.csv"
    bins_df.to_csv(bins_path, index=False)
    logger.info("Wrote %d rows to %s", len(bins_df), bins_path)

    return {
        "psem_path": str(psem_path),
        "pressure_bins_path": str(bins_path),
    }


# ============================================================================
# MAIN PIPELINE ENTRY POINT
# ============================================================================

def run_behavior_analysis_pipeline(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
    skip_detection: bool = False,
) -> Dict[str, Any]:
    """
    Run full behavior analysis pipeline.

    Steps:
    1. Detection and mapping (if not skipped)
    2. Behavioral metrics
    3. PSEM and bins outputs

    Args:
        output_root: Root output directory
        prompts_path: Path to prompts.csv
        skip_detection: If True, skip detection step (assumes already done)

    Returns:
        Dict with all output paths and stats
    """
    results = {}

    # Step 1: Detection and mapping
    if not skip_detection:
        logger.info("Step 1: Running detection and mapping...")
        detection_result = run_detection_mapping(output_root, prompts_path)
        results["detection"] = detection_result

    # Step 2: Behavioral metrics
    logger.info("Step 2: Computing behavioral metrics...")
    try:
        behavior_df = compute_behavioral_metrics(output_root, prompts_path)
        results["behavior_metrics_count"] = len(behavior_df)
    except FileNotFoundError as e:
        logger.warning("Skipping behavioral metrics: %s", e)

    # Step 3: PSEM and bins
    logger.info("Step 3: Writing PSEM and pressure bins...")
    psem_result = write_psem_and_bins(output_root, prompts_path)
    results["psem"] = psem_result

    logger.info("Behavior analysis pipeline complete.")
    return results


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("BEHAVIOR ANALYSIS MODULE TESTS")
    print("=" * 60)

    # Test 1: load_jsonl
    print("\n1. Testing load_jsonl (with missing file):")
    try:
        items = list(load_jsonl(Path("/nonexistent/file.jsonl")))
        print(f"   OK: empty generator for missing file ({len(items)} items)")
    except Exception as e:
        print(f"   FAIL: {e}")

    # Test 2: compute_suppression_metrics
    print("\n2. Testing compute_suppression_metrics:")
    try:
        metrics = compute_suppression_metrics(0.8, 0.2)
        print(f"   delta={metrics['delta']:.3f}, relative={metrics['relative']:.3f}")
        assert abs(metrics["delta"] - 0.6) < 0.001
        print("   OK")
    except Exception as e:
        print(f"   FAIL: {e}")

    print("\n" + "=" * 60)
    print("Tests complete.")
    print("=" * 60)
