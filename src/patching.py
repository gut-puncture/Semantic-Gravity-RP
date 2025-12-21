"""
patching.py - Activation Patching for Causal Intervention Analysis

This module provides:
- Subset selection for patching experiments
- Activation patching with four patch types (A/B/C/D)
- P_sem computation via first-token decomposition
- Pipeline entrypoint for running full patching analysis

Per specification Section 10 and execution-plan Section 10.
"""

import json
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS
# ============================================================================

try:
    from .utils import ModelWrapper, word_in_text, normalize_for_match
    from .prompt_builder import build_prompt
    from .config import get_base_paths, CONFIG
    from .metrics_psem import token_sequences_for_variants, compute_p_sem
    from .metrics_attn import (
        _compute_char_spans,
        _get_offsets_for_prompt,
        _map_char_span_to_token_indices,
    )
except ImportError:
    from utils import ModelWrapper, word_in_text, normalize_for_match
    from prompt_builder import build_prompt
    from config import get_base_paths, CONFIG
    from metrics_psem import token_sequences_for_variants, compute_p_sem
    from metrics_attn import (
        _compute_char_spans,
        _get_offsets_for_prompt,
        _map_char_span_to_token_indices,
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
) -> "pd.DataFrame":
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
    import pandas as pd

    path = prompts_path or (data_root / "prompts.csv")
    if not path.exists():
        raise FileNotFoundError(f"prompts.csv not found: {path}")

    df = pd.read_csv(path)

    required = ["prompt_id", "category", "question_text", "target_word", "p0", "p0_bin"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"prompts.csv missing columns: {missing}")

    return df


def _load_detection_mapping(run_root: Path) -> Dict[str, bool]:
    """
    Load detection_mapping.jsonl and extract outcome for negative greedy.

    Args:
        run_root: Run root directory

    Returns:
        Dict mapping prompt_id -> word_present (bool)

    Raises:
        FileNotFoundError: If detection_mapping.jsonl not found
    """
    path = run_root / "runs" / "detection_mapping.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"detection_mapping.jsonl not found: {path}. "
            "Run detection mapping first."
        )

    outcome_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Keep only negative greedy (sample_id == "")
                if record.get("condition") != "negative":
                    continue
                if record.get("sample_id", "") != "":
                    continue
                prompt_id = str(record.get("prompt_id", ""))
                word_present = record.get("word_present", False)
                outcome_map[prompt_id] = word_present
            except json.JSONDecodeError:
                continue

    return outcome_map


def _load_greedy_completions(run_root: Path) -> Dict[str, Dict]:
    """
    Load completions_greedy.jsonl for negative condition.

    Args:
        run_root: Run root directory

    Returns:
        Dict mapping prompt_id -> {generated_text, generated_token_ids}
    """
    path = run_root / "runs" / "completions_greedy.jsonl"
    if not path.exists():
        return {}

    completions = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("condition") != "negative":
                    continue
                prompt_id = str(record.get("prompt_id", ""))
                completions[prompt_id] = {
                    "generated_text": record.get("generated_text", ""),
                    "generated_token_ids": record.get("generated_token_ids", []),
                }
            except json.JSONDecodeError:
                continue

    return completions


# ============================================================================
# SUBSET SELECTION
# ============================================================================


def select_patching_subset(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Select stratified subset of prompts for patching.

    Uses CONFIG['patching'] for parameters:
    - prompts_per_bin: target per bin (default 50)
    - total_subset_size: target total (default 250)
    - min_failures: minimum failures required (default 30)

    Args:
        output_root: Run root directory
        prompts_path: Path to prompts.csv
        seed: Random seed for reproducibility

    Returns:
        List of selected prompt records
    """
    import pandas as pd

    run_root = _resolve_run_root(output_root)
    runs_dir = run_root / "runs"
    subset_path = runs_dir / "patching_subset.json"

    # Reuse existing subset if present
    if subset_path.exists():
        try:
            with open(subset_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing:
                logger.info("Reusing existing subset: %d prompts", len(existing))
                return existing
        except Exception as e:
            logger.warning("Failed to load existing subset: %s", e)

    paths = get_base_paths()
    data_root = paths.get("data_root", Path("data"))

    # Load data
    prompts_df = _load_prompts_df(prompts_path, data_root)
    outcome_map = _load_detection_mapping(run_root)

    # Get config
    patching_cfg = CONFIG.get("patching", {})
    prompts_per_bin = patching_cfg.get("prompts_per_bin", 50)
    min_failures = patching_cfg.get("min_failures", 30)

    if seed is None:
        seed = CONFIG.get("seeds", {}).get("python", 42)

    rng = random.Random(seed)

    # Join with outcomes
    prompts_df["outcome"] = prompts_df["prompt_id"].apply(
        lambda pid: "failure" if outcome_map.get(str(pid), False) else "success"
    )

    # Group by p0_bin
    bins = prompts_df["p0_bin"].unique()
    selected = []
    selected_ids: Set[str] = set()

    for bin_name in sorted(bins):
        bin_df = prompts_df[prompts_df["p0_bin"] == bin_name]

        failures = bin_df[bin_df["outcome"] == "failure"].to_dict("records")
        successes = bin_df[bin_df["outcome"] == "success"].to_dict("records")

        rng.shuffle(failures)
        rng.shuffle(successes)

        # Select failures first
        bin_selected = []
        if len(failures) <= prompts_per_bin:
            bin_selected.extend(failures)
        else:
            bin_selected.extend(failures[:prompts_per_bin])

        # Fill with successes if needed
        remaining = prompts_per_bin - len(bin_selected)
        if remaining > 0:
            bin_selected.extend(successes[:remaining])

        for rec in bin_selected:
            pid = str(rec["prompt_id"])
            if pid not in selected_ids:
                selected.append(rec)
                selected_ids.add(pid)

    # Check min_failures
    total_failures = sum(1 for r in selected if r["outcome"] == "failure")

    if total_failures < min_failures:
        # Gather remaining unselected failures
        all_failures = prompts_df[prompts_df["outcome"] == "failure"]
        extra_failures = [
            r for r in all_failures.to_dict("records")
            if str(r["prompt_id"]) not in selected_ids
        ]
        rng.shuffle(extra_failures)

        # Swap in for successes
        success_indices = [
            i for i, r in enumerate(selected) if r["outcome"] == "success"
        ]
        rng.shuffle(success_indices)

        swapped = 0
        for extra in extra_failures:
            if total_failures >= min_failures:
                break
            if not success_indices:
                break
            swap_idx = success_indices.pop()
            old_pid = str(selected[swap_idx]["prompt_id"])
            selected_ids.remove(old_pid)
            selected[swap_idx] = extra
            selected_ids.add(str(extra["prompt_id"]))
            total_failures += 1
            swapped += 1

        if total_failures < min_failures:
            logger.warning(
                "Only %d failures available (min required: %d)",
                total_failures, min_failures
            )

    # Prepare output records
    output_records = []
    for rec in selected:
        output_records.append({
            "prompt_id": str(rec["prompt_id"]),
            "category": rec["category"],
            "p0": float(rec["p0"]),
            "p0_bin": rec["p0_bin"],
            "outcome": rec["outcome"],
        })

    # Write subset
    runs_dir.mkdir(parents=True, exist_ok=True)
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=2)

    logger.info(
        "Selected %d prompts (%d failures) for patching subset",
        len(output_records), total_failures
    )

    return output_records


# ============================================================================
# CORE PATCHING UTILITIES
# ============================================================================


def _compute_p_rest_sum(
    token_sequences: List[Tuple[int, ...]],
    model: Any,
    tokenizer: Any,
    context_ids: List[int],
) -> Dict[int, float]:
    """
    Precompute suffix probability sums grouped by first token.

    For each first token t, compute:
    p_rest_sum[t] = sum over suffixes of P(suffix | context + [t])

    Args:
        token_sequences: List of target token sequences
        model: Language model
        tokenizer: Tokenizer
        context_ids: Context token IDs

    Returns:
        Dict mapping first_token_id -> p_rest_sum
    """
    import torch

    # Group by first token
    sequences_by_first: Dict[int, List[Tuple[int, ...]]] = defaultdict(list)
    for seq in token_sequences:
        if seq:
            first_tok = seq[0]
            suffix = seq[1:] if len(seq) > 1 else ()
            sequences_by_first[first_tok].append(suffix)

    p_rest_sum = {}

    for first_tok, suffixes in sequences_by_first.items():
        total = 0.0
        extended_context = context_ids + [first_tok]

        for suffix in suffixes:
            if not suffix:
                # Empty suffix means probability = 1.0
                total += 1.0
            else:
                # Compute P(suffix | context + first_tok)
                try:
                    p_suffix = compute_p_sem(model, tokenizer, extended_context, [suffix])
                    total += p_suffix
                except Exception:
                    pass

        p_rest_sum[first_tok] = total

    return p_rest_sum


def _compute_p_sem_from_logits(
    logits: "torch.Tensor",
    p_rest_sum: Dict[int, float],
) -> float:
    """
    Compute P_sem from logits using precomputed p_rest_sum.

    P_sem = sum_t p_first[t] * p_rest_sum[t]

    Args:
        logits: Logits tensor [vocab_size]
        p_rest_sum: Dict mapping first_token_id -> p_rest_sum

    Returns:
        P_sem value
    """
    import torch

    probs = torch.softmax(logits, dim=-1)
    p_sem = 0.0

    for first_tok, rest_sum in p_rest_sum.items():
        p_first = probs[first_tok].item()
        p_sem += p_first * rest_sum

    return min(max(p_sem, 0.0), 1.0)


def _generate_greedy_after_patch(
    model: Any,
    tokenizer: Any,
    first_token_id: int,
    past_key_values: Any,
    max_new_tokens: int = 8,
) -> str:
    """
    Generate greedy completion starting from patched first token.

    Args:
        model: Language model
        tokenizer: Tokenizer
        first_token_id: First token from patched logits
        past_key_values: KV cache from first token generation
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text string
    """
    import torch

    generated_ids = [first_token_id]
    current_past = past_key_values

    for _ in range(max_new_tokens - 1):
        next_input = torch.tensor([[generated_ids[-1]]], device=model.device)

        with torch.inference_mode():
            outputs = model(
                input_ids=next_input,
                past_key_values=current_past,
                use_cache=True,
            )

        next_logits = outputs.logits[0, -1, :]
        next_token = next_logits.argmax().item()
        generated_ids.append(next_token)
        current_past = outputs.past_key_values

        # Stop on EOS
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


# ============================================================================
# MAIN PATCHING FUNCTION
# ============================================================================


def run_activation_patching(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Path:
    """
    Run activation patching on selected subset.

    Implements four patch types per layer:
    - A: Full residual output
    - B: Attention module output
    - C: Instruction token KV
    - D: Question token KV

    Args:
        output_root: Run root directory
        prompts_path: Path to prompts.csv
        limit: Optional limit on prompts to process

    Returns:
        Path to patching_results.csv
    """
    import pandas as pd
    import torch

    run_root = _resolve_run_root(output_root)
    runs_dir = run_root / "runs"
    subset_path = runs_dir / "patching_subset.json"

    # Load subset
    if not subset_path.exists():
        raise FileNotFoundError(
            f"Patching subset not found: {subset_path}. "
            "Run select_patching_subset first."
        )

    with open(subset_path, "r", encoding="utf-8") as f:
        subset = json.load(f)

    if limit:
        subset = subset[:limit]

    paths = get_base_paths()
    data_root = paths.get("data_root", Path("data"))

    # Load prompts for question_text and target_word
    prompts_df = _load_prompts_df(prompts_path, data_root)
    prompt_info = {
        str(row["prompt_id"]): {
            "question_text": row["question_text"],
            "target_word": row["target_word"],
        }
        for _, row in prompts_df.iterrows()
    }

    # Load detection mapping and greedy completions
    outcome_map = _load_detection_mapping(run_root)
    greedy_completions = _load_greedy_completions(run_root)

    # Load model
    wrapper = ModelWrapper.get_instance()
    if not wrapper.is_loaded:
        wrapper.load()

    model = wrapper.model
    tokenizer = wrapper.tokenizer

    # Find transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise RuntimeError("Cannot find transformer layers in model")

    num_layers = len(layers)
    max_new_tokens = CONFIG.get("model", {}).get("max_new_tokens_greedy", 8)

    results = []

    for rec in subset:
        prompt_id = str(rec["prompt_id"])
        category = rec["category"]
        p0_bin = rec["p0_bin"]
        outcome = rec["outcome"]

        info = prompt_info.get(prompt_id)
        if not info:
            logger.warning("No prompt info for %s", prompt_id)
            continue

        question_text = info["question_text"]
        target_word = info["target_word"]

        # Build prompts
        baseline_text = build_prompt(question_text, target_word, "baseline")
        negative_text = build_prompt(question_text, target_word, "negative")

        # Tokenize
        baseline_inputs = tokenizer(baseline_text, return_tensors="pt")
        negative_inputs = tokenizer(negative_text, return_tensors="pt")

        baseline_ids = baseline_inputs["input_ids"].to(model.device)
        negative_ids = negative_inputs["input_ids"].to(model.device)

        # Split into prefix and last token
        if baseline_ids.shape[1] < 2 or negative_ids.shape[1] < 2:
            logger.warning("Prompt too short for %s", prompt_id)
            continue

        baseline_prefix = baseline_ids[:, :-1]
        baseline_last = baseline_ids[:, -1:]
        negative_prefix = negative_ids[:, :-1]
        negative_last = negative_ids[:, -1:]

        prefix_len = negative_prefix.shape[1]

        # Get token sequences for P_sem
        token_seqs = token_sequences_for_variants(target_word, tokenizer)
        context_ids_neg = tokenizer.encode(negative_text, add_special_tokens=False)

        # Compute p_rest_sum
        p_rest_sum = _compute_p_rest_sum(token_seqs, model, tokenizer, context_ids_neg)

        # Get span indices for NEGATIVE prompt
        neg_input_ids_list = negative_ids[0].tolist()
        neg_char_spans = _compute_char_spans(negative_text, target_word, "negative")
        neg_offsets = _get_offsets_for_prompt(negative_text, tokenizer, neg_input_ids_list)

        negative_instr_tokens = _map_char_span_to_token_indices(neg_offsets, neg_char_spans["instr_span"])
        negative_question_tokens = _map_char_span_to_token_indices(neg_offsets, neg_char_spans["question_span"])

        # Get span indices for BASELINE prompt
        base_input_ids_list = baseline_ids[0].tolist()
        base_char_spans = _compute_char_spans(baseline_text, target_word, "baseline")
        base_offsets = _get_offsets_for_prompt(baseline_text, tokenizer, base_input_ids_list)

        baseline_instr_tokens = _map_char_span_to_token_indices(base_offsets, base_char_spans["instr_span"])
        baseline_question_tokens = _map_char_span_to_token_indices(base_offsets, base_char_spans["question_span"])

        # Filter to respective prefix lengths
        baseline_prefix_len = baseline_prefix.shape[1]
        negative_prefix_len = negative_prefix.shape[1]

        negative_instr_tokens = [i for i in negative_instr_tokens if i < negative_prefix_len]
        negative_question_tokens = [i for i in negative_question_tokens if i < negative_prefix_len]
        baseline_instr_tokens = [i for i in baseline_instr_tokens if i < baseline_prefix_len]
        baseline_question_tokens = [i for i in baseline_question_tokens if i < baseline_prefix_len]

        # Align indices by relative order for KV patching
        instr_pairs = _align_span_indices(negative_instr_tokens, baseline_instr_tokens)
        question_pairs = _align_span_indices(negative_question_tokens, baseline_question_tokens)

        try:
            with torch.inference_mode():
                # Compute baseline prefix KV
                baseline_prefix_out = model(baseline_prefix, use_cache=True)
                baseline_past = baseline_prefix_out.past_key_values

                # Compute negative prefix KV
                negative_prefix_out = model(negative_prefix, use_cache=True)
                negative_past = negative_prefix_out.past_key_values

                # Capture baseline last-token activations
                baseline_residuals = {}
                baseline_attn_out = {}
                hooks = []

                for layer_idx, layer in enumerate(layers):
                    # Find attention module
                    attn_module = None
                    for attr in ["self_attn", "attention", "attn"]:
                        if hasattr(layer, attr):
                            attn_module = getattr(layer, attr)
                            break

                    if attn_module is not None:
                        def make_attn_hook(idx):
                            def hook(module, args, output):
                                if isinstance(output, tuple):
                                    baseline_attn_out[idx] = output[0][:, -1, :].detach().clone()
                                else:
                                    baseline_attn_out[idx] = output[:, -1, :].detach().clone()
                            return hook
                        hooks.append(attn_module.register_forward_hook(make_attn_hook(layer_idx)))

                baseline_last_out = model(
                    baseline_last,
                    past_key_values=baseline_past,
                    use_cache=True,
                    output_hidden_states=True,
                )

                # Remove hooks
                for h in hooks:
                    h.remove()
                hooks.clear()

                # Store residuals
                for layer_idx in range(num_layers):
                    baseline_residuals[layer_idx] = baseline_last_out.hidden_states[layer_idx + 1][:, -1, :].detach().clone()

                baseline_last_past = baseline_last_out.past_key_values

                # Compute negative unpatched
                negative_last_out = model(
                    negative_last,
                    past_key_values=negative_past,
                    use_cache=True,
                    output_hidden_states=True,
                )

                logits_neg = negative_last_out.logits[0, -1, :]
                p_sem_original = _compute_p_sem_from_logits(logits_neg, p_rest_sum)

                # Greedy token original
                original_violation = outcome_map.get(prompt_id, False)
                greedy_comp = greedy_completions.get(prompt_id, {})
                greedy_text_original = greedy_comp.get("generated_text", "")
                if not greedy_text_original:
                    greedy_token_original = tokenizer.decode([logits_neg.argmax().item()])
                else:
                    greedy_token_original = greedy_text_original.split()[0] if greedy_text_original.split() else ""

                negative_last_past = negative_last_out.past_key_values

                # Per-layer patching
                for layer_idx in range(num_layers):
                    layer = layers[layer_idx]

                    # Find attention module
                    attn_module = None
                    for attr in ["self_attn", "attention", "attn"]:
                        if hasattr(layer, attr):
                            attn_module = getattr(layer, attr)
                            break

                    # === Patch A: Residual ===
                    patched_residual = baseline_residuals[layer_idx]

                    def make_residual_hook(target_residual):
                        def hook(module, args, output):
                            if isinstance(output, tuple):
                                new_out = list(output)
                                new_out[0] = output[0].clone()
                                new_out[0][:, -1, :] = target_residual
                                return tuple(new_out)
                            else:
                                new_out = output.clone()
                                new_out[:, -1, :] = target_residual
                                return new_out
                        return hook

                    hook_a = layer.register_forward_hook(make_residual_hook(patched_residual))

                    try:
                        out_a = model(
                            negative_last,
                            past_key_values=negative_past,
                            use_cache=True,
                        )
                        logits_a = out_a.logits[0, -1, :]
                        p_sem_a = _compute_p_sem_from_logits(logits_a, p_rest_sum)
                        greedy_tok_a = logits_a.argmax().item()
                        patched_text_a = _generate_greedy_after_patch(
                            model, tokenizer, greedy_tok_a, out_a.past_key_values, max_new_tokens
                        )
                        patched_violation_a = word_in_text(target_word, patched_text_a)
                        flip_a = 1 if original_violation != patched_violation_a else 0
                    finally:
                        hook_a.remove()

                    results.append({
                        "prompt_id": prompt_id,
                        "category": category,
                        "p0_bin": p0_bin,
                        "outcome": outcome,
                        "layer": layer_idx,
                        "patch_type": "A",
                        "p_sem_original_neg": p_sem_original,
                        "p_sem_patched": p_sem_a,
                        "delta_p": p_sem_a - p_sem_original,
                        "flip_indicator": flip_a,
                        "greedy_token_original": greedy_token_original,
                        "greedy_token_patched": tokenizer.decode([greedy_tok_a]),
                    })

                    # === Patch B: Attention output ===
                    if attn_module is not None and layer_idx in baseline_attn_out:
                        patched_attn = baseline_attn_out[layer_idx]

                        def make_attn_patch_hook(target_attn):
                            def hook(module, args, output):
                                if isinstance(output, tuple):
                                    new_out = list(output)
                                    new_out[0] = output[0].clone()
                                    new_out[0][:, -1, :] = target_attn
                                    return tuple(new_out)
                                else:
                                    new_out = output.clone()
                                    new_out[:, -1, :] = target_attn
                                    return new_out
                            return hook

                        hook_b = attn_module.register_forward_hook(make_attn_patch_hook(patched_attn))

                        try:
                            out_b = model(
                                negative_last,
                                past_key_values=negative_past,
                                use_cache=True,
                            )
                            logits_b = out_b.logits[0, -1, :]
                            p_sem_b = _compute_p_sem_from_logits(logits_b, p_rest_sum)
                            greedy_tok_b = logits_b.argmax().item()
                            patched_text_b = _generate_greedy_after_patch(
                                model, tokenizer, greedy_tok_b, out_b.past_key_values, max_new_tokens
                            )
                            patched_violation_b = word_in_text(target_word, patched_text_b)
                            flip_b = 1 if original_violation != patched_violation_b else 0
                        finally:
                            hook_b.remove()

                        results.append({
                            "prompt_id": prompt_id,
                            "category": category,
                            "p0_bin": p0_bin,
                            "outcome": outcome,
                            "layer": layer_idx,
                            "patch_type": "B",
                            "p_sem_original_neg": p_sem_original,
                            "p_sem_patched": p_sem_b,
                            "delta_p": p_sem_b - p_sem_original,
                            "flip_indicator": flip_b,
                            "greedy_token_original": greedy_token_original,
                            "greedy_token_patched": tokenizer.decode([greedy_tok_b]),
                        })

                    # === Patch C: Instruction KV ===
                    if instr_pairs:
                        patched_past_c = _patch_kv_at_index_pairs(
                            negative_past, baseline_past, layer_idx, instr_pairs
                        )

                        out_c = model(
                            negative_last,
                            past_key_values=patched_past_c,
                            use_cache=True,
                        )
                        logits_c = out_c.logits[0, -1, :]
                        p_sem_c = _compute_p_sem_from_logits(logits_c, p_rest_sum)
                        greedy_tok_c = logits_c.argmax().item()
                        patched_text_c = _generate_greedy_after_patch(
                            model, tokenizer, greedy_tok_c, out_c.past_key_values, max_new_tokens
                        )
                        patched_violation_c = word_in_text(target_word, patched_text_c)
                        flip_c = 1 if original_violation != patched_violation_c else 0

                        results.append({
                            "prompt_id": prompt_id,
                            "category": category,
                            "p0_bin": p0_bin,
                            "outcome": outcome,
                            "layer": layer_idx,
                            "patch_type": "C",
                            "p_sem_original_neg": p_sem_original,
                            "p_sem_patched": p_sem_c,
                            "delta_p": p_sem_c - p_sem_original,
                            "flip_indicator": flip_c,
                            "greedy_token_original": greedy_token_original,
                            "greedy_token_patched": tokenizer.decode([greedy_tok_c]),
                        })

                    # === Patch D: Question KV ===
                    if question_pairs:
                        patched_past_d = _patch_kv_at_index_pairs(
                            negative_past, baseline_past, layer_idx, question_pairs
                        )

                        out_d = model(
                            negative_last,
                            past_key_values=patched_past_d,
                            use_cache=True,
                        )
                        logits_d = out_d.logits[0, -1, :]
                        p_sem_d = _compute_p_sem_from_logits(logits_d, p_rest_sum)
                        greedy_tok_d = logits_d.argmax().item()
                        patched_text_d = _generate_greedy_after_patch(
                            model, tokenizer, greedy_tok_d, out_d.past_key_values, max_new_tokens
                        )
                        patched_violation_d = word_in_text(target_word, patched_text_d)
                        flip_d = 1 if original_violation != patched_violation_d else 0

                        results.append({
                            "prompt_id": prompt_id,
                            "category": category,
                            "p0_bin": p0_bin,
                            "outcome": outcome,
                            "layer": layer_idx,
                            "patch_type": "D",
                            "p_sem_original_neg": p_sem_original,
                            "p_sem_patched": p_sem_d,
                            "delta_p": p_sem_d - p_sem_original,
                            "flip_indicator": flip_d,
                            "greedy_token_original": greedy_token_original,
                            "greedy_token_patched": tokenizer.decode([greedy_tok_d]),
                        })

        except Exception as e:
            logger.error("Patching failed for %s: %s", prompt_id, e)
            continue
        finally:
            torch.cuda.empty_cache()

    # Write results
    output_path = runs_dir / "patching_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d patching results to %s", len(df), output_path)

    return output_path


def _patch_kv_at_indices(
    neg_past: Tuple,
    baseline_past: Tuple,
    layer_idx: int,
    token_indices: List[int],
    prefix_len: int,
) -> Tuple:
    """
    Patch KV cache at specific token indices for one layer.

    Args:
        neg_past: Negative past_key_values
        baseline_past: Baseline past_key_values
        layer_idx: Layer to patch
        token_indices: Token indices to patch
        prefix_len: Prefix length for validation

    Returns:
        Patched past_key_values tuple
    """
    import torch

    # Convert to list for modification
    patched = []
    for i, layer_kv in enumerate(neg_past):
        if i == layer_idx:
            # layer_kv is (key, value) tuple
            k, v = layer_kv
            k_patched = k.clone()
            v_patched = v.clone()

            # Detect KV shape: [B, H, T, D] or [B, T, H, D]
            # T is the sequence length dimension
            if k.shape[2] == prefix_len:
                # Shape is [B, H, T, D]
                seq_dim = 2
            elif k.shape[1] == prefix_len:
                # Shape is [B, T, H, D]
                seq_dim = 1
            else:
                # Best guess: use dim 2
                seq_dim = 2

            base_k, base_v = baseline_past[i]

            for idx in token_indices:
                if idx >= min(k.shape[seq_dim], base_k.shape[seq_dim]):
                    continue
                if seq_dim == 2:
                    k_patched[:, :, idx, :] = base_k[:, :, idx, :]
                    v_patched[:, :, idx, :] = base_v[:, :, idx, :]
                else:
                    k_patched[:, idx, :, :] = base_k[:, idx, :, :]
                    v_patched[:, idx, :, :] = base_v[:, idx, :, :]

            patched.append((k_patched, v_patched))
        else:
            patched.append(layer_kv)

    return tuple(patched)


def _align_span_indices(
    negative_indices: List[int],
    baseline_indices: List[int],
) -> List[Tuple[int, int]]:
    """
    Align span indices between negative and baseline prompts by relative order.

    Returns pairs of (neg_idx, base_idx) that correspond to the same
    relative position within their respective spans.

    Args:
        negative_indices: Token indices from negative prompt span
        baseline_indices: Token indices from baseline prompt span

    Returns:
        List of (neg_idx, base_idx) pairs
    """
    if not negative_indices or not baseline_indices:
        return []
    return list(zip(negative_indices, baseline_indices))


def _patch_kv_at_index_pairs(
    neg_past: Tuple,
    baseline_past: Tuple,
    layer_idx: int,
    index_pairs: List[Tuple[int, int]],
) -> Tuple:
    """
    Patch KV cache using aligned (neg_idx, base_idx) pairs for one layer.

    For each pair, copies baseline KV at base_idx into negative KV at neg_idx.

    Args:
        neg_past: Negative past_key_values
        baseline_past: Baseline past_key_values
        layer_idx: Layer to patch
        index_pairs: List of (neg_idx, base_idx) pairs

    Returns:
        Patched past_key_values tuple
    """
    import torch

    if not index_pairs:
        return neg_past

    # Compute max indices needed
    max_neg_idx = max(p[0] for p in index_pairs)
    max_base_idx = max(p[1] for p in index_pairs)

    def detect_seq_dim(tensor, max_idx):
        """Detect which dimension is the sequence dimension based on max index."""
        if tensor.dim() == 4:
            # Shape is [B, ?, ?, D] - seq dim is 1 or 2
            dim1_size = tensor.shape[1]
            dim2_size = tensor.shape[2]
            # Pick dim where size > max_idx; prefer larger if both pass
            dim1_ok = dim1_size > max_idx
            dim2_ok = dim2_size > max_idx
            if dim1_ok and dim2_ok:
                # Both valid, pick larger (more likely seq dim)
                return 1 if dim1_size > dim2_size else 2
            elif dim2_ok:
                return 2
            elif dim1_ok:
                return 1
            else:
                logger.warning(
                    "Neither dim 1 (size %d) nor dim 2 (size %d) > max_idx %d; defaulting to 2",
                    dim1_size, dim2_size, max_idx
                )
                return 2
        else:
            logger.warning("KV tensor has dim %d (expected 4); defaulting seq_dim=2", tensor.dim())
            return 2

    patched = []
    for i, layer_kv in enumerate(neg_past):
        if i == layer_idx:
            k, v = layer_kv
            k_patched = k.clone()
            v_patched = v.clone()

            base_k, base_v = baseline_past[i]

            neg_seq_dim = detect_seq_dim(k, max_neg_idx)
            base_seq_dim = detect_seq_dim(base_k, max_base_idx)

            for neg_idx, base_idx in index_pairs:
                # Validate bounds
                if neg_idx >= k.shape[neg_seq_dim]:
                    continue
                if base_idx >= base_k.shape[base_seq_dim]:
                    continue

                # Extract base slice: always [B, H, D] or [B, D, H] depending on layout
                # Assign directly without transpose - both have same head/dim structure
                if base_seq_dim == 2:
                    base_k_slice = base_k[:, :, base_idx, :]
                    base_v_slice = base_v[:, :, base_idx, :]
                else:  # base_seq_dim == 1
                    base_k_slice = base_k[:, base_idx, :, :]
                    base_v_slice = base_v[:, base_idx, :, :]

                if neg_seq_dim == 2:
                    k_patched[:, :, neg_idx, :] = base_k_slice
                    v_patched[:, :, neg_idx, :] = base_v_slice
                else:  # neg_seq_dim == 1
                    k_patched[:, neg_idx, :, :] = base_k_slice
                    v_patched[:, neg_idx, :, :] = base_v_slice

            patched.append((k_patched, v_patched))
        else:
            patched.append(layer_kv)

    return tuple(patched)


# ============================================================================
# PIPELINE ENTRYPOINT
# ============================================================================


def run_activation_patching_pipeline(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run full activation patching pipeline.

    Steps:
    1. Select subset (reuses existing if present)
    2. Run patching

    Args:
        output_root: Run root directory
        prompts_path: Path to prompts.csv
        limit: Optional limit on prompts

    Returns:
        Dict with paths and counts
    """
    results = {}

    logger.info("Step 1: Selecting patching subset...")
    try:
        subset = select_patching_subset(output_root, prompts_path)
        results["subset_count"] = len(subset)
        results["subset_failures"] = sum(1 for r in subset if r["outcome"] == "failure")
    except Exception as e:
        logger.error("Subset selection failed: %s", e)
        results["subset_error"] = str(e)
        return results

    logger.info("Step 2: Running activation patching...")
    try:
        patching_path = run_activation_patching(output_root, prompts_path, limit)
        results["patching_results_path"] = str(patching_path)
    except Exception as e:
        logger.error("Patching failed: %s", e)
        results["patching_error"] = str(e)

    logger.info("Activation patching pipeline complete.")
    return results


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PATCHING MODULE SELF-TEST")
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

    # Test 2: _compute_p_sem_from_logits (mock)
    print("\n2. Testing _compute_p_sem_from_logits:")
    try:
        import torch
        mock_logits = torch.zeros(100)
        mock_logits[5] = 10.0  # High prob for token 5
        mock_p_rest = {5: 0.8, 10: 0.1}
        p_sem = _compute_p_sem_from_logits(mock_logits, mock_p_rest)
        print(f"   P_sem = {p_sem:.4f}")
        assert p_sem > 0.5, "Expected P_sem > 0.5 given high logit for token 5"
        print("   PASS: P_sem computation works")
    except ImportError:
        print("   SKIP: torch not available")
    except Exception as e:
        print(f"   FAIL: {e}")

    # Test 3: _patch_kv_at_indices (mock)
    print("\n3. Testing _patch_kv_at_indices:")
    try:
        import torch
        # Mock KV cache: 2 layers, each with (key, value) of shape [1, 4, 10, 8]
        neg_past = tuple(
            (torch.zeros(1, 4, 10, 8), torch.zeros(1, 4, 10, 8))
            for _ in range(2)
        )
        baseline_past = tuple(
            (torch.ones(1, 4, 10, 8), torch.ones(1, 4, 10, 8))
            for _ in range(2)
        )
        patched = _patch_kv_at_indices(neg_past, baseline_past, 0, [2, 3], 10)
        # Check that layer 0 positions 2,3 are patched
        k_patched = patched[0][0]
        assert k_patched[0, 0, 2, 0] == 1.0, "Position 2 should be patched"
        assert k_patched[0, 0, 0, 0] == 0.0, "Position 0 should not be patched"
        print("   PASS: KV patching works")
    except ImportError:
        print("   SKIP: torch not available")
    except Exception as e:
        print(f"   FAIL: {e}")

    # Test 4: _align_span_indices
    print("\n4. Testing _align_span_indices:")
    try:
        pairs = _align_span_indices([5, 6, 7], [2, 3, 4])
        assert pairs == [(5, 2), (6, 3), (7, 4)], f"Expected paired indices, got {pairs}"
        empty_pairs = _align_span_indices([], [1, 2])
        assert empty_pairs == [], "Empty input should return empty list"
        print("   PASS: Index alignment works")
    except Exception as e:
        print(f"   FAIL: {e}")

    # Test 5: _patch_kv_at_index_pairs with mismatched indices
    print("\n5. Testing _patch_kv_at_index_pairs:")
    try:
        import torch
        # neg KV shape [1, 4, 12, 8], baseline [1, 4, 10, 8]
        neg_past = tuple(
            (torch.zeros(1, 4, 12, 8), torch.zeros(1, 4, 12, 8))
            for _ in range(2)
        )
        baseline_past = tuple(
            (torch.ones(1, 4, 10, 8), torch.ones(1, 4, 10, 8))
            for _ in range(2)
        )
        # Patch neg_idx=5 from base_idx=2, neg_idx=6 from base_idx=3
        pairs = [(5, 2), (6, 3)]
        patched = _patch_kv_at_index_pairs(neg_past, baseline_past, 0, pairs)
        k_patched = patched[0][0]
        assert k_patched[0, 0, 5, 0] == 1.0, "neg_idx=5 should be patched from base_idx=2"
        assert k_patched[0, 0, 6, 0] == 1.0, "neg_idx=6 should be patched from base_idx=3"
        assert k_patched[0, 0, 2, 0] == 0.0, "neg_idx=2 should NOT be patched"
        assert k_patched[0, 0, 0, 0] == 0.0, "neg_idx=0 should NOT be patched"
        print("   PASS: Index-pair KV patching works")
    except ImportError:
        print("   SKIP: torch not available")
    except Exception as e:
        print(f"   FAIL: {e}")

    # Test 6: _patch_kv_at_index_pairs with mixed dimensions (neg [B,T,H,D], base [B,H,T,D])
    print("\n6. Testing _patch_kv_at_index_pairs mixed dimensions:")
    try:
        import torch
        # neg KV shape [1, 12, 4, 8] (seq_dim=1), baseline [1, 4, 10, 8] (seq_dim=2)
        neg_past = tuple(
            (torch.zeros(1, 12, 4, 8), torch.zeros(1, 12, 4, 8))
            for _ in range(2)
        )
        baseline_past = tuple(
            (torch.ones(1, 4, 10, 8), torch.ones(1, 4, 10, 8))
            for _ in range(2)
        )
        pairs = [(5, 2), (6, 3)]
        patched = _patch_kv_at_index_pairs(neg_past, baseline_past, 0, pairs)
        k_patched = patched[0][0]
        # neg is [B, T, H, D] so check k_patched[0, 5, 0, 0]
        assert k_patched[0, 5, 0, 0] == 1.0, "neg_idx=5 should be patched"
        assert k_patched[0, 6, 0, 0] == 1.0, "neg_idx=6 should be patched"
        assert k_patched[0, 2, 0, 0] == 0.0, "neg_idx=2 should NOT be patched"
        print("   PASS: Mixed dimension patching works")
    except ImportError:
        print("   SKIP: torch not available")
    except Exception as e:
        print(f"   FAIL: {e}")

    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)
