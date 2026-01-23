"""
behavioral_replication.py - Behavioral replication runner for multiple HF models.

Runs baseline semantic pressure (P_sem) and negative-instruction sampling for
arbitrary Hugging Face causal LMs. Produces per-model outputs without any
mechanistic interpretability steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime
    np = None
    pd = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled at runtime
    plt = None

logger = logging.getLogger(__name__)

try:
    from .config import CONFIG, get_base_paths, validate_environment
    from .prompt_builder import build_prompt
    from .utils import ModelWrapper, set_all_seeds
    from .metrics_psem import compute_p_sem_for_prompts, token_sequences_for_variants
    from .runner import _iter_batches, _tokenize_batch, _trim_generated_ids, load_existing_completions, append_completion
    from .behavior_analysis import run_detection_mapping, compute_behavioral_metrics
    from .bootstrap import _compute_violation_rate_ci
    from .visualize import _plot_violation_rate_vs_p0
except ImportError:
    from config import CONFIG, get_base_paths, validate_environment
    from prompt_builder import build_prompt
    from utils import ModelWrapper, set_all_seeds
    from metrics_psem import compute_p_sem_for_prompts, token_sequences_for_variants
    from runner import _iter_batches, _tokenize_batch, _trim_generated_ids, load_existing_completions, append_completion
    from behavior_analysis import run_detection_mapping, compute_behavioral_metrics
    from bootstrap import _compute_violation_rate_ci
    from visualize import _plot_violation_rate_vs_p0


@dataclass
class ReplicationRunConfig:
    model_id: str
    prompts_path: Path
    output_root: Optional[Path] = None
    run_tag: Optional[str] = None
    num_samples: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    behavioral_batch_size: int = 4
    prompt_batch_size: int = 32
    task_batch_size: int = 64
    max_batch_tokens: Optional[int] = None
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 42
    logistic_bootstrap: int = 200


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for behavioral replication.")


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("numpy is required for behavioral replication.")


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for behavioral replication.")


def _slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(" ", "_")


def _resolve_run_root(output_root: Optional[Path], run_tag: Optional[str], model_id: str) -> Path:
    if output_root is None:
        base_output = get_base_paths().get("output_root", Path("outputs"))
    else:
        base_output = Path(output_root)
        base_output.mkdir(parents=True, exist_ok=True)

    slug = _slugify_model_id(model_id)
    if run_tag is not None:
        run_root = base_output / f"experiment_run_{run_tag}_{slug}"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M")
        run_root = base_output / f"experiment_run_{timestamp}_{slug}"

    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "runs").mkdir(parents=True, exist_ok=True)
    (run_root / "figures").mkdir(parents=True, exist_ok=True)
    (run_root / "errors").mkdir(parents=True, exist_ok=True)
    return run_root


def _load_prompts_df(prompts_path: Path) -> "pd.DataFrame":
    _require_pandas()
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts.csv not found: {prompts_path}")
    df = pd.read_csv(prompts_path)
    required = ["prompt_id", "target_word", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"prompts.csv missing columns: {missing}")
    df["prompt_id"] = df["prompt_id"].astype(str)
    return df


def _get_prompt_texts(df: "pd.DataFrame") -> Tuple[List[str], List[str]]:
    baseline_texts: List[str] = []
    negative_texts: List[str] = []

    has_prompt_text = "prompt_text" in df.columns
    has_negative_text = "negative_prompt_text" in df.columns

    for _, row in df.iterrows():
        question_text = row.get("question_text", "")
        target_word = row.get("target_word", "")
        prompt_text = row.get("prompt_text") if has_prompt_text else None
        neg_text = row.get("negative_prompt_text") if has_negative_text else None

        if isinstance(prompt_text, str) and prompt_text.strip():
            baseline = prompt_text
        else:
            if not question_text:
                raise ValueError("Missing question_text for prompt without prompt_text.")
            baseline = build_prompt(question_text, target_word, "baseline")

        if isinstance(neg_text, str) and neg_text.strip():
            negative = neg_text
        else:
            if not question_text:
                raise ValueError("Missing question_text for prompt without negative_prompt_text.")
            negative = build_prompt(question_text, target_word, "negative")

        baseline_texts.append(baseline)
        negative_texts.append(negative)

    return baseline_texts, negative_texts


def _compute_p0_bins(p0_values: List[float]) -> List[str]:
    pressure_bins = CONFIG.get("dataset", {}).get("pressure_bins", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins: List[str] = []

    for p0 in p0_values:
        p0_clamped = max(0.0, min(1.0, float(p0)))
        assigned = False
        for i in range(len(pressure_bins) - 1):
            lower = pressure_bins[i]
            upper = pressure_bins[i + 1]
            if i == len(pressure_bins) - 2:
                in_bin = lower <= p0_clamped <= upper
            else:
                in_bin = lower <= p0_clamped < upper
            if in_bin:
                bins.append(f"{lower:.1f}-{upper:.1f}")
                assigned = True
                break
        if not assigned:
            bins.append(f"{pressure_bins[-2]:.1f}-{pressure_bins[-1]:.1f}")
    return bins


def _write_run_metadata(run_root: Path, model_id: str, prompts_path: Path, config: ReplicationRunConfig) -> None:
    metadata = validate_environment()
    metadata.update({
        "model_id": model_id,
        "prompts_path": str(prompts_path),
        "num_samples": config.num_samples,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "behavioral_batch_size": config.behavioral_batch_size,
        "prompt_batch_size": config.prompt_batch_size,
        "task_batch_size": config.task_batch_size,
        "max_batch_tokens": config.max_batch_tokens,
        "bootstrap_iterations": config.bootstrap_iterations,
        "bootstrap_seed": config.bootstrap_seed,
        "logistic_bootstrap": config.logistic_bootstrap,
    })
    path = run_root / "run_metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)


def _compute_p0_values(
    wrapper: ModelWrapper,
    baseline_texts: List[str],
    target_words: List[str],
    prompt_batch_size: int,
    task_batch_size: int,
    max_batch_tokens: Optional[int],
    log_every: int = 50,
) -> List[float]:
    total = len(baseline_texts)
    p0_values: List[float] = []

    start = time.monotonic()
    last_log = 0

    for i in range(0, total, max(1, int(prompt_batch_size))):
        batch_texts = baseline_texts[i:i + prompt_batch_size]
        batch_targets = target_words[i:i + prompt_batch_size]
        p_sem = compute_p_sem_for_prompts(
            prompt_texts=batch_texts,
            target_words=batch_targets,
            model=wrapper.model,
            tokenizer=wrapper.tokenizer,
            batch_size=max(1, int(task_batch_size)),
            max_batch_tokens=max_batch_tokens,
            strict=True,
            logger=logger,
        )
        p0_values.extend(p_sem)

        completed = len(p0_values)
        if log_every and (completed - last_log) >= log_every:
            elapsed = max(time.monotonic() - start, 1e-6)
            rate = completed / elapsed
            remaining = total - completed
            eta = remaining / rate if rate > 0 else 0.0
            logger.info(
                "P0 progress: %d/%d (%.1f%%) | %.2f prompts/s | ETA %.1fs",
                completed,
                total,
                100.0 * completed / total,
                rate,
                eta,
            )
            last_log = completed

    return p0_values


def _write_prompts_model_csv(
    df: "pd.DataFrame",
    p0_values: List[float],
    run_root: Path,
) -> Path:
    p0_bins = _compute_p0_bins(p0_values)

    out_df = df.copy()
    out_df["p0"] = p0_values
    out_df["p1"] = float("nan")
    out_df["p0_bin"] = p0_bins

    out_path = run_root / "runs" / "prompts_model.csv"
    out_df.to_csv(out_path, index=False)

    logger.info("Wrote per-model prompts CSV to %s", out_path)
    return out_path


def _write_psem_and_bins(
    df: "pd.DataFrame",
    p0_values: List[float],
    run_root: Path,
    wrapper: ModelWrapper,
) -> None:
    _require_pandas()
    psem_path = run_root / "runs" / "psem.csv"
    bins_path = run_root / "runs" / "pressure_bins.csv"

    token_seq_cache: Dict[str, int] = {}
    num_sequences: List[int] = []
    for word in df["target_word"].tolist():
        if word in token_seq_cache:
            num_sequences.append(token_seq_cache[word])
            continue
        seqs = token_sequences_for_variants(word, wrapper.tokenizer)
        token_seq_cache[word] = len(seqs)
        num_sequences.append(len(seqs))

    psem_rows = {
        "prompt_id": df["prompt_id"].astype(str),
        "category": df.get("category", ""),
        "p0": p0_values,
        "p1": [float("nan")] * len(p0_values),
        "delta": [float("nan")] * len(p0_values),
        "relative": [float("nan")] * len(p0_values),
        "log": [float("nan")] * len(p0_values),
        "num_token_sequences": num_sequences,
    }
    pd.DataFrame(psem_rows).to_csv(psem_path, index=False)

    pressure_bins = CONFIG.get("dataset", {}).get("pressure_bins", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    p0_bins = _compute_p0_bins(p0_values)
    bin_id_map = {f"{pressure_bins[i]:.1f}-{pressure_bins[i+1]:.1f}": i for i in range(len(pressure_bins) - 1)}
    bin_lower_map = {f"{pressure_bins[i]:.1f}-{pressure_bins[i+1]:.1f}": pressure_bins[i] for i in range(len(pressure_bins) - 1)}
    bin_upper_map = {f"{pressure_bins[i]:.1f}-{pressure_bins[i+1]:.1f}": pressure_bins[i + 1] for i in range(len(pressure_bins) - 1)}

    bins_rows = {
        "prompt_id": df["prompt_id"].astype(str),
        "category": df.get("category", ""),
        "p0": p0_values,
        "bin_id": [bin_id_map.get(b, len(pressure_bins) - 2) for b in p0_bins],
        "bin_lower": [bin_lower_map.get(b, pressure_bins[-2]) for b in p0_bins],
        "bin_upper": [bin_upper_map.get(b, pressure_bins[-1]) for b in p0_bins],
        "p0_bin": p0_bins,
    }
    pd.DataFrame(bins_rows).to_csv(bins_path, index=False)

    logger.info("Wrote psem.csv to %s", psem_path)
    logger.info("Wrote pressure_bins.csv to %s", bins_path)


def _generate_negative_samples(
    wrapper: ModelWrapper,
    prompt_ids: List[str],
    negative_texts: List[str],
    run_root: Path,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    log_every: int = 200,
) -> None:
    import torch

    completions_path = run_root / "runs" / "completions_samples.jsonl"
    existing = load_existing_completions(completions_path)

    tasks: List[Dict[str, Any]] = []
    for prompt_id, prompt_text in zip(prompt_ids, negative_texts):
        sample_ids = []
        for sample_id in range(num_samples):
            key = f"{prompt_id}|negative|{sample_id}"
            if key not in existing:
                sample_ids.append(sample_id)
        if sample_ids:
            tasks.append({
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "samples_to_run": sample_ids,
            })

    if not tasks:
        logger.info("All negative samples already exist; skipping generation.")
        return

    sample_groups: Dict[int, List[Dict[str, Any]]] = {}
    for task in tasks:
        sample_groups.setdefault(len(task["samples_to_run"]), []).append(task)

    total_samples = sum(len(t["samples_to_run"]) for t in tasks)
    processed = 0
    last_log = 0
    start = time.monotonic()

    def _process_batch(batch: List[Dict[str, Any]], num_to_generate: int) -> None:
        nonlocal processed, last_log

        prompt_texts = [t["prompt_text"] for t in batch]
        tokenized = _tokenize_batch(wrapper, prompt_texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        batch_input_len = input_ids.shape[1]

        with torch.inference_mode():
            outputs = wrapper.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_to_generate,
                pad_token_id=wrapper.tokenizer.pad_token_id,
            )

        eos_token_id = wrapper.tokenizer.eos_token_id
        pad_token_id = wrapper.tokenizer.pad_token_id
        timestamp = time.time()

        for i, task in enumerate(batch):
            prompt_id = task["prompt_id"]
            sample_ids = task["samples_to_run"]

            for j, sample_id in enumerate(sample_ids):
                seq_idx = i * num_to_generate + j
                seq = outputs[seq_idx]
                generated_ids = seq[batch_input_len:].tolist()
                generated_ids = _trim_generated_ids(generated_ids, eos_token_id, pad_token_id)
                generated_text = wrapper.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                finish_reason = "stop" if len(generated_ids) < max_new_tokens else "length"
                record = {
                    "prompt_id": prompt_id,
                    "condition": "negative",
                    "sample_id": sample_id,
                    "generated_text": generated_text,
                    "generated_token_ids": generated_ids,
                    "finish_reason": finish_reason,
                    "timestamp": timestamp,
                }
                append_completion(completions_path, record)
                existing.add(f"{prompt_id}|negative|{sample_id}")
                processed += 1

        del outputs
        del input_ids
        del attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if log_every and (processed - last_log) >= log_every:
            elapsed = max(time.monotonic() - start, 1e-6)
            rate = processed / elapsed
            remaining = total_samples - processed
            eta = remaining / rate if rate > 0 else 0.0
            logger.info(
                "Negative sampling: %d/%d (%.1f%%) | %.2f samples/s | ETA %.1fs",
                processed,
                total_samples,
                100.0 * processed / max(1, total_samples),
                rate,
                eta,
            )
            last_log = processed

    def _run_batch(batch: List[Dict[str, Any]], num_to_generate: int) -> None:
        try:
            _process_batch(batch, num_to_generate)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            logger.warning("OOM in batch of %d prompts; splitting", len(batch))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch) <= 1:
                raise
            mid = len(batch) // 2
            _run_batch(batch[:mid], num_to_generate)
            _run_batch(batch[mid:], num_to_generate)

    for num_to_generate, group in sample_groups.items():
        if num_to_generate <= 0:
            continue
        for batch in _iter_batches(group, max(1, int(batch_size))):
            _run_batch(batch, num_to_generate)

    logger.info("Negative sampling complete: %d samples", processed)


def _compute_violation_bootstrap(
    prompts_df: "pd.DataFrame",
    behavior_df: "pd.DataFrame",
    run_root: Path,
    n_iterations: int,
    seed: int,
) -> Path:
    _require_numpy()
    rng = np.random.default_rng(seed)
    ci_percentiles = (2.5, 97.5)
    results = _compute_violation_rate_ci(prompts_df, behavior_df, rng, n_iterations, ci_percentiles, seed)
    bootstrap_path = run_root / "runs" / "bootstrap_results.csv"
    pd.DataFrame(results).to_csv(bootstrap_path, index=False)
    logger.info("Wrote violation bootstrap results to %s", bootstrap_path)
    return bootstrap_path


def _plot_violation_bins(bootstrap_path: Path, run_root: Path) -> Optional[Path]:
    _require_matplotlib()
    df = pd.read_csv(bootstrap_path)
    figures_dir = run_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_violation_rate_vs_p0(df, figures_dir)
    plot_path = figures_dir / "violation_rate_vs_p0.png"
    return plot_path


def _fit_logistic(
    detection_path: Path,
    p0_by_id: Dict[str, float],
    figures_dir: Path,
    n_bootstrap: int,
) -> Optional[Dict[str, Any]]:
    _require_numpy()
    _require_matplotlib()

    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    rows = []
    with detection_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("condition") != "negative":
                continue
            if rec.get("sample_id", "") == "":
                continue
            pid = str(rec.get("prompt_id"))
            if pid not in p0_by_id:
                continue
            rows.append({
                "prompt_id": pid,
                "p0": float(p0_by_id[pid]),
                "violation": 1 if rec.get("word_present", False) else 0,
            })

    if not rows:
        logger.warning("No samples found for logistic regression.")
        return None

    samples_df = pd.DataFrame(rows)
    X = samples_df[["p0"]].values
    y = samples_df["violation"].values

    if len(np.unique(y)) < 2:
        logger.warning("Logistic regression skipped: only one class present.")
        return None

    log_model = LogisticRegression(solver="lbfgs", max_iter=1000)
    log_model.fit(X, y)
    coef = float(log_model.coef_[0, 0])
    intercept = float(log_model.intercept_[0])

    grid = np.linspace(0.0, 1.0, 101)
    logit_pred = log_model.predict_proba(grid.reshape(-1, 1))[:, 1]

    rng = np.random.default_rng(42)
    boot_preds = []
    boot_coefs = []
    for _ in range(int(n_bootstrap)):
        boot = samples_df.sample(n=len(samples_df), replace=True, random_state=int(rng.integers(1e9)))
        yb = boot["violation"].values
        if len(np.unique(yb)) < 2:
            continue
        model_b = LogisticRegression(solver="lbfgs", max_iter=1000)
        model_b.fit(boot[["p0"]].values, yb)
        boot_preds.append(model_b.predict_proba(grid.reshape(-1, 1))[:, 1])
        boot_coefs.append((float(model_b.intercept_[0]), float(model_b.coef_[0, 0])))

    if boot_preds:
        boot_preds_arr = np.array(boot_preds)
        ci_low, ci_high = np.percentile(boot_preds_arr, [2.5, 97.5], axis=0)
    else:
        ci_low = None
        ci_high = None

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(samples_df["p0"].values, samples_df["violation"].values)
    iso_pred = iso.predict(grid)

    plt.figure(figsize=(7, 5))
    plt.scatter(samples_df["p0"], samples_df["violation"], s=6, alpha=0.05, label="samples")
    plt.plot(grid, logit_pred, color="red", label="logistic fit")
    if ci_low is not None and ci_high is not None:
        plt.fill_between(grid, ci_low, ci_high, color="red", alpha=0.2, label="logistic 95% CI")
    plt.plot(grid, iso_pred, color="black", linestyle="--", label="isotonic fit")
    plt.xlabel("P0 (pressure)")
    plt.ylabel("Violation probability")
    plt.title("Violation vs P0: logistic + isotonic fits")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = figures_dir / "violation_rate_fit.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    result = {
        "coef": coef,
        "intercept": intercept,
        "n_samples": len(samples_df),
        "bootstraps": len(boot_preds),
    }

    if boot_coefs:
        vals = np.array(boot_coefs)
        intercept_ci = np.percentile(vals[:, 0], [2.5, 97.5])
        coef_ci = np.percentile(vals[:, 1], [2.5, 97.5])
        result.update({
            "intercept_ci_low": float(intercept_ci[0]),
            "intercept_ci_high": float(intercept_ci[1]),
            "coef_ci_low": float(coef_ci[0]),
            "coef_ci_high": float(coef_ci[1]),
        })

    return result


def run_behavioral_replication(config: ReplicationRunConfig) -> Dict[str, Any]:
    _require_pandas()
    _require_numpy()

    set_all_seeds()

    model_cfg = CONFIG.get("model", {})
    num_samples = config.num_samples or int(model_cfg.get("num_stochastic_samples", 16))
    max_new_tokens = config.max_new_tokens or int(model_cfg.get("max_new_tokens_stochastic", 10))
    temperature = config.temperature if config.temperature is not None else float(model_cfg.get("temperature", 1.0))
    top_p = config.top_p if config.top_p is not None else float(model_cfg.get("top_p", 0.9))

    run_root = _resolve_run_root(config.output_root, config.run_tag, config.model_id)
    _write_run_metadata(run_root, config.model_id, config.prompts_path, config)

    logger.info("Loading prompts from %s", config.prompts_path)
    prompts_df = _load_prompts_df(config.prompts_path)
    baseline_texts, negative_texts = _get_prompt_texts(prompts_df)

    wrapper = ModelWrapper.get_instance()
    if wrapper.is_loaded:
        wrapper.unload()
    wrapper.load(config.model_id, force_reload=True)

    logger.info("Computing baseline pressure (P0)...")
    p0_values = _compute_p0_values(
        wrapper=wrapper,
        baseline_texts=baseline_texts,
        target_words=prompts_df["target_word"].tolist(),
        prompt_batch_size=config.prompt_batch_size,
        task_batch_size=config.task_batch_size,
        max_batch_tokens=config.max_batch_tokens,
    )

    prompts_model_path = _write_prompts_model_csv(prompts_df, p0_values, run_root)
    _write_psem_and_bins(prompts_df, p0_values, run_root, wrapper)

    logger.info("Generating negative-instruction samples...")
    _generate_negative_samples(
        wrapper=wrapper,
        prompt_ids=prompts_df["prompt_id"].astype(str).tolist(),
        negative_texts=negative_texts,
        run_root=run_root,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=config.behavioral_batch_size,
    )

    logger.info("Running detection mapping...")
    run_detection_mapping(output_root=run_root, prompts_path=prompts_model_path)

    logger.info("Computing behavioral metrics...")
    behavior_df = compute_behavioral_metrics(output_root=run_root, prompts_path=prompts_model_path)

    bootstrap_path = _compute_violation_bootstrap(
        prompts_df=pd.read_csv(prompts_model_path),
        behavior_df=behavior_df,
        run_root=run_root,
        n_iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
    )
    _plot_violation_bins(bootstrap_path, run_root)

    figures_dir = run_root / "figures"
    detection_path = run_root / "runs" / "detection_mapping.jsonl"
    p0_by_id = dict(zip(prompts_df["prompt_id"].astype(str), p0_values))
    logistic_result = _fit_logistic(
        detection_path=detection_path,
        p0_by_id=p0_by_id,
        figures_dir=figures_dir,
        n_bootstrap=config.logistic_bootstrap,
    )

    summary = {
        "model_id": config.model_id,
        "run_root": str(run_root),
        "n_prompts": int(len(prompts_df)),
        "mean_violation_rate": float(behavior_df["violation_rate"].mean()) if not behavior_df.empty else float("nan"),
        "num_samples_per_prompt": num_samples,
        "logistic_fit": logistic_result,
    }

    summary_path = run_root / "runs" / "behavior_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    wrapper.unload()

    logger.info("Replication run complete: %s", run_root)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run behavioral replication on HF models.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model IDs (HF hub IDs or local paths).",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to prompts.csv (2500 prompts).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Base output directory (defaults to configured outputs).",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag (used as experiment_run_<tag>).",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Samples per prompt (default from config).")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens (default from config).")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p for sampling.")
    parser.add_argument("--behavioral-batch-size", type=int, default=4, help="Batch size for sampling.")
    parser.add_argument("--prompt-batch-size", type=int, default=32, help="Prompt batch size for P0.")
    parser.add_argument("--task-batch-size", type=int, default=64, help="Task batch size for P0.")
    parser.add_argument("--max-batch-tokens", type=int, default=None, help="Max tokens per P0 batch.")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000, help="Bootstrap iterations.")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Bootstrap seed.")
    parser.add_argument("--logistic-bootstrap", type=int, default=200, help="Logistic regression bootstraps.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    summaries = []
    prompts_path = Path(args.prompts)
    output_root = Path(args.output_root) if args.output_root else None

    for model_id in args.models:
        cfg = ReplicationRunConfig(
            model_id=model_id,
            prompts_path=prompts_path,
            output_root=output_root,
            run_tag=args.run_tag,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            behavioral_batch_size=args.behavioral_batch_size,
            prompt_batch_size=args.prompt_batch_size,
            task_batch_size=args.task_batch_size,
            max_batch_tokens=args.max_batch_tokens,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
            logistic_bootstrap=args.logistic_bootstrap,
        )
        summary = run_behavioral_replication(cfg)
        summaries.append(summary)

    if summaries:
        base_output = output_root or get_base_paths().get("output_root", Path("outputs"))
        summary_path = Path(base_output) / "behavioral_replication_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=True)
        logger.info("Wrote summary table to %s", summary_path)


if __name__ == "__main__":
    main()
