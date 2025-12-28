"""
metrics_attn.py - Mechanistic Interpretability Metrics

This module provides:
- Attention metrics from mechanistic traces (IAR, NF, TMF, PI)
- Logit lens analysis across layers
- Attention/FFN decomposition for activation patching prep

Per specification Section 9 and execution-plan Section 10.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS
# ============================================================================

try:
    from .utils import ModelWrapper, compute_token_char_spans, resolve_run_root
    from .prompt_builder import build_prompt
    from .config import get_base_paths, CONFIG
    from .metrics_psem import token_sequences_for_variants
except ImportError:
    from utils import ModelWrapper, compute_token_char_spans, resolve_run_root
    from prompt_builder import build_prompt
    from config import get_base_paths, CONFIG
    from metrics_psem import token_sequences_for_variants

# Optional import for debug
try:
    try:
        from .utils import normalize_for_match
    except ImportError:
        from utils import normalize_for_match
except ImportError:
    normalize_for_match = None


# ============================================================================
# HELPER: RESOLVE RUN ROOT
# ============================================================================


def _resolve_run_root(output_root: Optional[Path]) -> Path:
    """
    Resolve the run root directory from output_root or find the latest run.

    NOTE: This is a local alias for utils.resolve_run_root for backward compatibility.
    New code should import resolve_run_root from utils directly.
    """
    return resolve_run_root(output_root)


# ============================================================================
# HELPER: DETECTION MAPPING LOADING
# ============================================================================


def _load_detection_mapping(runs_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Load detection_mapping_greedy.jsonl as a lookup dict keyed by (prompt_id, condition).
    
    Uses greedy-only detection file to avoid collision with behavioral samples.
    Falls back to detection_mapping.jsonl if greedy-only not found.
    Returns empty dict if neither file exists.
    """
    # Prefer greedy-only file to avoid collision with behavioral samples
    greedy_path = runs_dir / "detection_mapping_greedy.jsonl"
    fallback_path = runs_dir / "detection_mapping.jsonl"
    
    mapping_path = greedy_path if greedy_path.exists() else fallback_path
    
    result: Dict[Tuple[str, str], Dict] = {}
    
    if not mapping_path.exists():
        logger.warning("No detection mapping found at %s or %s", greedy_path, fallback_path)
        return result
    
    if mapping_path == fallback_path and greedy_path != fallback_path:
        logger.warning(
            "Using fallback %s - consider generating greedy-only file", 
            fallback_path
        )
    
    try:
        with open(mapping_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    key = (str(entry.get("prompt_id", "")), str(entry.get("condition", "")))
                    result[key] = entry
    except Exception as e:
        logger.error("Failed to load detection mapping: %s", e)
    
    return result


def _get_decision_step(
    prompt_id: str,
    condition: str,
    detection_mapping: Dict[Tuple[str, str], Dict],
    generated_len: int,
) -> Tuple[Optional[int], bool, bool]:
    """
    Get the decision step for attention/logit lens computation.
    
    The decision step is the step (within generated sequence) where the target
    token is about to be emitted. This is the position for measuring internal
    signals like attention and logit lens.
    
    Args:
        prompt_id: The prompt identifier
        condition: "baseline" or "negative"
        detection_mapping: Loaded detection mapping dict
        generated_len: Length of generated sequence (for validation)
    
    Returns:
        Tuple of:
        - decision_step: Index within attentions list (0 = first generated token).
          For word_present=False (obey case), returns 0.
          Returns None only if entry missing or mapping_error=True.
        - word_present: Whether target word was detected in completion
        - mapping_error: Whether there was a mapping error (skip this entry)
    
    Logic:
    - If target is first generated token: decision_step = 0
    - If target starts at token T: decision_step = T
    - If target not present (obey case): decision_step = 0
    - If mapping_error=True: return (None, word_present, True) to skip
    """
    key = (prompt_id, condition)
    entry = detection_mapping.get(key)
    
    if entry is None:
        # No detection entry - return None to indicate fallback needed
        return (None, False, False)
    
    word_present = entry.get("word_present", False)
    mapping_error = entry.get("mapping_error", False)
    
    # Fix C: If mapping_error, skip this entry entirely
    if mapping_error:
        return (None, word_present, True)
    
    # Fix B: For word_present=False (obey case), use step 0
    # We still need to measure metrics for success cases
    if not word_present:
        return (0, False, False)
    
    pre_target_indices = entry.get("pre_target_token_indices", [])
    if not pre_target_indices:
        # Target detected but no valid token span mapping
        # Fallback to step 0 (first generated step)
        return (0, True, False)
    
    # Use first occurrence's pre-target index (spec: first occurrence only)
    pre_idx = pre_target_indices[0]
    
    if pre_idx is None:
        # Target is the first generated token - use step 0
        return (0, True, False)
    
    # pre_idx is the index (0-indexed within generated) of token before target
    # Target first token is at index = pre_idx + 1
    # decision_step = target token index = pre_idx + 1
    decision_step = pre_idx + 1
    
    # Validate bounds
    if decision_step < 0 or decision_step >= generated_len:
        logger.debug(
            "Decision step %d out of bounds for generated_len %d (prompt=%s)",
            decision_step, generated_len, prompt_id
        )
        # Clamp to valid range
        decision_step = max(0, min(decision_step, generated_len - 1))
    
    return (decision_step, True, False)


def _get_metric_target_ids(
    word_present: bool,
    exact_first_ids: List[int],
    detection_entry: Optional[Dict],
    generated_ids_list: Optional[List[int]],
    decision_step: int,
    tokenizer: Any,
) -> Tuple[List[int], str]:
    """
    Compute the metric target set (first token IDs) for logit lens/decomp.
    
    Handles context-aware tracking: if target appeared as multi-token split,
    track the actual prefix token used instead of full exact set.
    
    Args:
        word_present: Whether target was detected
        exact_first_ids: Precomputed first IDs from token_sequences_for_variants
        detection_entry: Detection mapping entry for this prompt/condition
        generated_ids_list: List of generated token IDs
        decision_step: Computed decision step
        tokenizer: Tokenizer for encoding variants
    
    Returns:
        Tuple of:
        - metric_target_ids: List of token IDs to track
        - metric_target_kind: "exact" or "prefix"
    """
    # Default: use exact set
    if not word_present:
        return (exact_first_ids, "exact")
    
    # Check token span length from detection mapping
    token_spans = detection_entry.get("token_spans", []) if detection_entry else []
    if not token_spans:
        return (exact_first_ids, "exact")
    
    # Use first occurrence only
    first_span = token_spans[0]
    if not isinstance(first_span, (list, tuple)) or len(first_span) < 2:
        return (exact_first_ids, "exact")
    
    span_start, span_end = first_span[0], first_span[1]
    span_length = span_end - span_start + 1 if span_end >= span_start else 1
    
    # Single-token: use exact set
    if span_length == 1:
        return (exact_first_ids, "exact")
    
    # Multi-token (split failure): track prefix token
    # Use span_start from detection mapping (more reliable than decision_step)
    if not generated_ids_list:
        return (exact_first_ids, "exact")
    if span_start is None or span_start < 0 or span_start >= len(generated_ids_list):
        return (exact_first_ids, "exact")
    
    actual_prefix_id = generated_ids_list[span_start]
    prefix_ids = {actual_prefix_id}
    
    # Build prefix variants (capitalize, lowercase, etc.)
    prefix_text = tokenizer.decode([actual_prefix_id])
    
    # Preserve leading space
    has_leading_space = prefix_text.startswith(" ") or prefix_text.startswith("\u0120")
    base_text = prefix_text.lstrip()
    if has_leading_space:
        space_prefix = prefix_text[:len(prefix_text) - len(base_text)]
    else:
        space_prefix = ""
    
    # Capitalization variants
    variants = [
        base_text,
        base_text.lower(),
        base_text.upper(),
        base_text.capitalize(),
    ]
    
    for variant in variants:
        full_variant = space_prefix + variant
        encoded = tokenizer.encode(full_variant, add_special_tokens=False)
        if len(encoded) == 1:
            prefix_ids.add(encoded[0])
    
    return (sorted(prefix_ids), "prefix")

# ============================================================================
# HELPER: PROMPT SPAN EXTRACTION (CHARACTER SPANS)
# ============================================================================

# Prompt templates (must match prompt_builder.py)
_BASELINE_PREFIX = "Answer with exactly one English word."
_NEGATIVE_LINE_TEMPLATE = 'Do not use the word "{target}" anywhere in your answer.'


def _compute_char_spans(prompt_text: str, target_word: str, condition: str) -> dict:
    """
    Compute character spans for key prompt regions.

    Args:
        prompt_text: The full prompt text
        target_word: The target word X
        condition: "baseline" or "negative"

    Returns:
        Dict with:
        - instr_span: (start, end) or None
        - negation_span: (start, end) or None
        - target_mention_span: (start, end) or None
        - question_span: (start, end) or None
    """
    result = {
        "instr_span": None,
        "negation_span": None,
        "target_mention_span": None,
        "question_span": None,
    }

    # Locate question span (common to both conditions)
    q_marker = "Question: "
    a_marker = "\nAnswer:"
    q_start_idx = prompt_text.find(q_marker)
    if q_start_idx >= 0:
        q_content_start = q_start_idx + len(q_marker)
        a_idx = prompt_text.find(a_marker, q_content_start)
        if a_idx >= 0:
            result["question_span"] = (q_content_start, a_idx)
        else:
            # Fallback: question goes to end
            result["question_span"] = (q_content_start, len(prompt_text))

    if condition == "baseline":
        # instr_span = "Answer with exactly one English word."
        idx = prompt_text.find(_BASELINE_PREFIX)
        if idx >= 0:
            result["instr_span"] = (idx, idx + len(_BASELINE_PREFIX))
        # negation_span and target_mention_span remain None

    elif condition == "negative":
        # Build the expected prohibition line
        prohibition_line = _NEGATIVE_LINE_TEMPLATE.format(target=target_word)

        # Try exact match first
        idx = prompt_text.find(prohibition_line)
        if idx >= 0:
            result["instr_span"] = (idx, idx + len(prohibition_line))

            # negation_span: "Do not use" within prohibition line
            neg_phrase = "Do not use"
            neg_idx = prompt_text.find(neg_phrase, idx)
            if neg_idx >= 0 and neg_idx < idx + len(prohibition_line):
                result["negation_span"] = (neg_idx, neg_idx + len(neg_phrase))

            # target_mention_span: the quoted target word
            # Find the target within quotes in the prohibition line
            quoted_target = f'"{target_word}"'
            tgt_idx = prompt_text.find(quoted_target, idx)
            if tgt_idx >= 0 and tgt_idx < idx + len(prohibition_line):
                # Span is just the word, not the quotes
                result["target_mention_span"] = (tgt_idx + 1, tgt_idx + 1 + len(target_word))

        else:
            # Fallback: search for "Do not use" marker
            neg_phrase = "Do not use"
            neg_idx = prompt_text.find(neg_phrase)
            if neg_idx >= 0:
                # Find end of line
                line_end = prompt_text.find("\n", neg_idx)
                if line_end < 0:
                    line_end = len(prompt_text)
                result["instr_span"] = (neg_idx, line_end)
                result["negation_span"] = (neg_idx, neg_idx + len(neg_phrase))

                # Try to find target in that region
                quoted_target = f'"{target_word}"'
                tgt_idx = prompt_text.find(quoted_target, neg_idx)
                if tgt_idx >= 0 and tgt_idx < line_end:
                    result["target_mention_span"] = (tgt_idx + 1, tgt_idx + 1 + len(target_word))

    return result


# ============================================================================
# HELPER: MAP CHAR SPANS TO TOKEN INDICES
# ============================================================================


def _map_char_span_to_token_indices(
    offsets: List[Tuple[int, int]],
    span: Optional[Tuple[int, int]],
) -> List[int]:
    """
    Map a character span to overlapping token indices.

    Args:
        offsets: List of (start, end) character offsets for each token
        span: (start, end) character span or None

    Returns:
        List of token indices that overlap with the span
    """
    if span is None:
        return []

    span_start, span_end = span
    indices = []

    for i, offset in enumerate(offsets):
        if offset is None:
            continue
        if offset == (0, 0):
            continue
        tok_start, tok_end = offset
        # Token overlaps if: tok_end > span_start AND tok_start < span_end
        if tok_end > span_start and tok_start < span_end:
            indices.append(i)

    return indices


def _get_offsets_for_prompt(
    prompt_text: str,
    tokenizer: Any,
    input_ids: List[int],
) -> List[Tuple[int, int]]:
    """
    Get character offsets for tokens in prompt.

    Uses CONFIG['model']['add_special_tokens'] for consistency with generation.
    Falls back to incremental decode if offset_mapping unavailable.

    Args:
        prompt_text: The prompt text string
        tokenizer: Tokenizer instance
        input_ids: List of token IDs

    Returns:
        List of (start, end) character offsets for each token
    """
    # Use CONFIG setting for consistency with generation
    add_special_tokens = CONFIG.get("model", {}).get("add_special_tokens", False)

    # Try tokenizer's return_offsets_mapping with consistent setting
    try:
        encoded = tokenizer(
            prompt_text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping", [])
        if len(offsets) == len(input_ids):
            return offsets
        else:
            logger.debug(
                "Offset mapping length mismatch: %d vs %d input_ids",
                len(offsets), len(input_ids)
            )
    except Exception as e:
        logger.debug("Tokenizer offset_mapping failed: %s", e)

    # Fallback to incremental decode (always consistent)
    try:
        offsets = compute_token_char_spans(input_ids, tokenizer)
        return offsets
    except Exception as e:
        logger.warning("Failed to compute token char spans: %s", e)
        return [(0, 0)] * len(input_ids)


# ============================================================================
# ATTENTION METRICS FROM MECHANISTIC TRACES
# ============================================================================


def compute_attention_metrics(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Path:
    """
    Compute attention-based metrics from mechanistic traces.

    Metrics computed per layer/head at the TARGET DECISION STEP
    (the step where the target token is about to be emitted):
    - IAR: Instruction Attention Ratio
    - NF: Negation Focus (within instruction span)
    - TMF: Target Mention Focus (within instruction span)
    - PI: Polarity Index (TMF - NF)

    NOTE: Uses detection_mapping.jsonl to determine the correct decision step.
    If target was not detected, metrics are skipped for that completion.

    Args:
        output_root: Run root directory. If None, auto-selects latest.
        prompts_path: Path to prompts.csv. If None, uses data_root.
        limit: Optional limit on number of prompts to process.

    Returns:
        Path to attention_metrics.csv
    """
    import pandas as pd
    import torch

    # Resolve paths
    run_root = _resolve_run_root(output_root)
    runs_dir = run_root / "runs"
    trace_dir = runs_dir / "mechanistic_trace"

    paths = get_base_paths()
    data_root = paths.get("data_root", Path("data"))
    prompts_path = prompts_path or (data_root / "prompts.csv")

    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts.csv not found: {prompts_path}")

    # Load prompts
    prompts_df = pd.read_csv(prompts_path)

    required_cols = ["prompt_id", "question_text", "target_word"]
    missing = [c for c in required_cols if c not in prompts_df.columns]
    if missing:
        raise ValueError(f"prompts.csv missing columns: {missing}")

    if limit:
        prompts_df = prompts_df.head(limit)

    # Load detection mapping for decision step lookup
    detection_mapping = _load_detection_mapping(runs_dir)
    if not detection_mapping:
        logger.warning(
            "Detection mapping not found - falling back to step 0 for all. "
            "Run detection first for correct measurement."
        )

    # Get tokenizer
    wrapper = ModelWrapper.get_instance()
    if wrapper.tokenizer is None:
        wrapper.load()
    tokenizer = wrapper.tokenizer

    rows = []
    skipped_mapping_error = 0
    skipped_no_entry = 0

    for _, row in prompts_df.iterrows():
        prompt_id = str(row["prompt_id"])
        question_text = row["question_text"]
        target_word = row["target_word"]

        for condition in ["baseline", "negative"]:
            trace_path = trace_dir / f"{prompt_id}_{condition}.pt"

            if not trace_path.exists():
                logger.debug("Trace not found: %s", trace_path)
                continue

            try:
                trace = torch.load(trace_path, map_location="cpu")
            except Exception as e:
                logger.warning("Failed to load trace %s: %s", trace_path, e)
                continue

            # Check for attentions
            attentions = trace.get("attentions")
            if not attentions or len(attentions) == 0:
                logger.debug("No attentions in trace: %s", trace_path)
                continue

            # Get input_ids for context length
            input_ids = trace.get("input_ids")
            if input_ids is None:
                continue

            if hasattr(input_ids, "tolist"):
                if input_ids.dim() > 1:
                    input_ids_list = input_ids[0].tolist()
                else:
                    input_ids_list = input_ids.tolist()
            else:
                input_ids_list = list(input_ids)

            context_len = len(input_ids_list)
            generated_len = len(attentions)

            # Get decision step from detection mapping
            decision_step, word_present, mapping_error = _get_decision_step(
                prompt_id, condition, detection_mapping, generated_len
            )

            # Fix C: Skip entries with mapping_error
            if mapping_error:
                skipped_mapping_error += 1
                continue

            # If no detection entry exists (decision_step is None and not mapping_error),
            # use fallback step 0
            if decision_step is None:
                if not detection_mapping:
                    # No detection mapping at all - use step 0 as fallback
                    decision_step = 0
                    word_present = False
                else:
                    # Entry missing for this prompt - skip
                    skipped_no_entry += 1
                    continue

            # Validate step is in bounds
            if decision_step >= len(attentions):
                logger.debug(
                    "Decision step %d >= len(attentions) %d for %s_%s, using last step",
                    decision_step, len(attentions), prompt_id, condition
                )
                decision_step = len(attentions) - 1

            # Build prompt and get spans
            prompt_text = build_prompt(question_text, target_word, condition)
            char_spans = _compute_char_spans(prompt_text, target_word, condition)

            # Get offsets and map to token indices
            offsets = _get_offsets_for_prompt(prompt_text, tokenizer, input_ids_list)

            instr_tokens = _map_char_span_to_token_indices(offsets, char_spans["instr_span"])
            negation_tokens = _map_char_span_to_token_indices(offsets, char_spans["negation_span"])
            target_tokens = _map_char_span_to_token_indices(offsets, char_spans["target_mention_span"])
            question_tokens = _map_char_span_to_token_indices(offsets, char_spans["question_span"])

            # Use decision step attentions (NOT step 0)
            step_attn = attentions[decision_step]  # Tuple of layer tensors

            num_layers = len(step_attn)
            all_head_metrics = []

            for layer_idx, layer_attn in enumerate(step_attn):
                # layer_attn shape: [1, num_heads, seq_len, seq_len]
                # For the decision step, we look at the generated position attending to context
                if layer_attn.dim() < 4:
                    continue

                num_heads = layer_attn.shape[1]
                seq_len = layer_attn.shape[2]
                # The position attending = context_len + decision_step (current position in full seq)
                gen_pos = seq_len - 1  # Last position in this step's attention

                for head_idx in range(num_heads):
                    # Attention weights from gen_pos to all context positions
                    attn_weights = layer_attn[0, head_idx, gen_pos, :context_len].float()

                    # Compute mass for each span
                    def mass(token_indices):
                        if not token_indices:
                            return 0.0
                        valid_indices = [i for i in token_indices if i < len(attn_weights)]
                        if not valid_indices:
                            return 0.0
                        return attn_weights[valid_indices].sum().item()

                    mass_instr = mass(instr_tokens)
                    mass_negation = mass(negation_tokens)
                    mass_target = mass(target_tokens)
                    mass_question = mass(question_tokens)

                    # Compute metrics
                    iar = mass_instr / (mass_instr + mass_question + 1e-9)
                    nf = mass_negation / (mass_instr + 1e-9)
                    tmf = mass_target / (mass_instr + 1e-9)
                    pi = tmf - nf

                    head_row = {
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "decision_step": decision_step,  # Added for visibility
                        "word_present": word_present,
                        "layer": layer_idx,
                        "head": head_idx,
                        "iar": iar,
                        "nf": nf,
                        "tmf": tmf,
                        "pi": pi,
                        "aggregate_flag": "head",
                    }
                    rows.append(head_row)
                    all_head_metrics.append((layer_idx, head_idx, iar, nf, tmf, pi))

                # Per-layer mean
                layer_heads = [(h, iar, nf, tmf, pi) for (l, h, iar, nf, tmf, pi) in all_head_metrics if l == layer_idx]
                if layer_heads:
                    mean_iar = sum(x[1] for x in layer_heads) / len(layer_heads)
                    mean_nf = sum(x[2] for x in layer_heads) / len(layer_heads)
                    mean_tmf = sum(x[3] for x in layer_heads) / len(layer_heads)
                    mean_pi = sum(x[4] for x in layer_heads) / len(layer_heads)

                    rows.append({
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "decision_step": decision_step,
                        "word_present": word_present,
                        "layer": layer_idx,
                        "head": -1,
                        "iar": mean_iar,
                        "nf": mean_nf,
                        "tmf": mean_tmf,
                        "pi": mean_pi,
                        "aggregate_flag": "layer_mean",
                    })

            # Global mean
            if all_head_metrics:
                global_iar = sum(x[2] for x in all_head_metrics) / len(all_head_metrics)
                global_nf = sum(x[3] for x in all_head_metrics) / len(all_head_metrics)
                global_tmf = sum(x[4] for x in all_head_metrics) / len(all_head_metrics)
                global_pi = sum(x[5] for x in all_head_metrics) / len(all_head_metrics)

                rows.append({
                    "prompt_id": prompt_id,
                    "condition": condition,
                    "decision_step": decision_step,
                    "word_present": word_present,
                    "layer": -1,
                    "head": -1,
                    "iar": global_iar,
                    "nf": global_nf,
                    "tmf": global_tmf,
                    "pi": global_pi,
                    "aggregate_flag": "global_mean",
                })

    if skipped_mapping_error > 0:
        logger.info("Skipped %d completions due to mapping_error", skipped_mapping_error)
    if skipped_no_entry > 0:
        logger.info("Skipped %d completions with missing detection entries", skipped_no_entry)

    # Write output
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_path = runs_dir / "attention_metrics.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d attention metric rows to %s", len(df), output_path)

    return output_path


# ============================================================================
# LOGIT LENS + ATTN/FFN DECOMPOSITION
# ============================================================================


def compute_logit_lens_and_decomp(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Compute logit lens and attention/FFN decomposition at TARGET DECISION STEP.

    For each prompt/condition where target was detected:
    - Builds extended prefix: prompt + generated tokens up to (not including) target
    - Runs forward pass with hooks to capture layer activations at decision step
    - Computes p_sem at each layer (logit lens)
    - Decomposes into attention and FFN contributions

    NOTE: Uses detection_mapping.jsonl and completion traces to determine correct step.
    If target was not detected, skips the entry.

    Args:
        output_root: Run root directory. If None, auto-selects latest.
        prompts_path: Path to prompts.csv. If None, uses data_root.
        limit: Optional limit on number of prompts to process.

    Returns:
        Dict with paths to logit_lens.csv and ffn_attn_decomp.csv
    """
    import pandas as pd
    import torch

    # Resolve paths
    run_root = _resolve_run_root(output_root)
    runs_dir = run_root / "runs"
    trace_dir = runs_dir / "mechanistic_trace"

    paths = get_base_paths()
    data_root = paths.get("data_root", Path("data"))
    prompts_path = prompts_path or (data_root / "prompts.csv")

    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts.csv not found: {prompts_path}")

    # Load prompts
    prompts_df = pd.read_csv(prompts_path)

    required_cols = ["prompt_id", "question_text", "target_word"]
    missing = [c for c in required_cols if c not in prompts_df.columns]
    if missing:
        raise ValueError(f"prompts.csv missing columns: {missing}")

    if limit:
        prompts_df = prompts_df.head(limit)

    # Load detection mapping for decision step lookup
    detection_mapping = _load_detection_mapping(runs_dir)
    if not detection_mapping:
        logger.warning(
            "Detection mapping not found - will compute at prompt-end (last token). "
            "Run detection first for correct measurement."
        )

    # Load model/tokenizer
    wrapper = ModelWrapper.get_instance()
    if not wrapper.is_loaded:
        wrapper.load()

    model = wrapper.model
    tokenizer = wrapper.tokenizer

    # Find transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        final_norm = getattr(model.model, "norm", None)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        final_norm = getattr(model.transformer, "ln_f", None)
    else:
        raise RuntimeError("Cannot find transformer layers in model")

    num_layers = len(layers)
    lm_head = model.lm_head if hasattr(model, "lm_head") else None

    if lm_head is None:
        raise RuntimeError("Cannot find lm_head in model")

    # Data collection
    logit_lens_rows = []
    decomp_rows = []
    skipped_mapping_error = 0
    skipped_no_entry = 0
    skipped_missing_trace = 0

    for _, row in prompts_df.iterrows():
        prompt_id = str(row["prompt_id"])
        question_text = row["question_text"]
        target_word = row["target_word"]

    # Get first token IDs for target (for logit lens p_sem calculation)
    # Exact set = only single-token variants (avoid ambiguous prefixes)
    seqs = token_sequences_for_variants(target_word, tokenizer)
    first_ids = sorted({seq[0] for seq in seqs if len(seq) == 1})

        for condition in ["baseline", "negative"]:
            trace_path = trace_dir / f"{prompt_id}_{condition}.pt"

            # Load trace to get generated tokens
            generated_ids_list = None
            greedy_token_id = None
            if trace_path.exists():
                try:
                    trace = torch.load(trace_path, map_location="cpu")
                    gen_ids = trace.get("generated_ids")
                    if gen_ids is not None:
                        if hasattr(gen_ids, "tolist"):
                            if gen_ids.dim() > 1:
                                generated_ids_list = gen_ids[0].tolist()
                            else:
                                generated_ids_list = gen_ids.tolist()
                        elif isinstance(gen_ids, list):
                            generated_ids_list = gen_ids
                        if generated_ids_list:
                            greedy_token_id = generated_ids_list[0]
                except Exception as e:
                    logger.debug("Failed to load trace for %s_%s: %s", prompt_id, condition, e)

            # Get decision step from detection mapping
            gen_len = len(generated_ids_list) if generated_ids_list else 1
            decision_step, word_present, mapping_error = _get_decision_step(
                prompt_id, condition, detection_mapping, gen_len
            )

            # Fix C: Skip entries with mapping_error
            if mapping_error:
                skipped_mapping_error += 1
                continue

            # If no detection entry exists, use fallback or skip
            if decision_step is None:
                if not detection_mapping:
                    decision_step = 0
                    word_present = False
                else:
                    skipped_no_entry += 1
                    continue

            # Fix F: If decision_step > 0 but we don't have enough generated tokens, we cannot
            # construct the correct prefix - skip this entry rather than computing at wrong position
            # Note: We need generated_ids_list[:decision_step] for prefix AND generated_ids_list[decision_step] for greedy token
            if decision_step > 0 and (not generated_ids_list or len(generated_ids_list) <= decision_step):
                logger.warning(
                    "Skipping %s_%s: decision_step=%d but only %d generated tokens available",
                    prompt_id, condition, decision_step, 
                    len(generated_ids_list) if generated_ids_list else 0
                )
                skipped_missing_trace += 1
                continue

            # Build extended prefix: prompt tokens + generated tokens up to decision step
            prompt_text = build_prompt(question_text, target_word, condition)
            add_special_tokens = CONFIG.get("model", {}).get("add_special_tokens", False)
            prompt_inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=add_special_tokens)
            prompt_ids = prompt_inputs["input_ids"][0].tolist()

            # decision_step is the index of the token being generated
            # We need prefix = prompt + generated[0:decision_step]
            if decision_step > 0:
                prefix_ids = prompt_ids + generated_ids_list[:decision_step]
                # Fix E: greedy_token_id should be the token AT decision_step (not token 0)
                greedy_token_id = generated_ids_list[decision_step]
            else:
                # decision_step = 0 means first token, so prefix = just prompt
                prefix_ids = prompt_ids
                # greedy_token_id is already set to generated_ids_list[0] or will be computed

            # Context-aware metric target set (Change 1)
            # Get detection entry for token span info
            detection_key = (prompt_id, condition)
            detection_entry = detection_mapping.get(detection_key)
            
            metric_target_ids, metric_target_kind = _get_metric_target_ids(
                word_present=word_present,
                exact_first_ids=first_ids,
                detection_entry=detection_entry,
                generated_ids_list=generated_ids_list,
                decision_step=decision_step,
                tokenizer=tokenizer,
            )

            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=model.device)
            decision_position = len(prefix_ids) - 1  # Last position in extended prefix

            # Storage for hook captures
            h_in_storage = {}
            attn_out_storage = {}
            ffn_out_storage = {}
            hooks = []

            # Register hooks
            for layer_idx, layer in enumerate(layers):
                # Pre-hook for layer input
                def make_pre_hook(idx):
                    def hook(module, args):
                        if isinstance(args, tuple) and len(args) > 0:
                            h_in_storage[idx] = args[0].detach()
                        return None
                    return hook

                hooks.append(layer.register_forward_pre_hook(make_pre_hook(layer_idx)))

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
                                attn_out_storage[idx] = output[0].detach()
                            else:
                                attn_out_storage[idx] = output.detach()
                        return hook

                    hooks.append(attn_module.register_forward_hook(make_attn_hook(layer_idx)))

                # Find FFN module
                ffn_module = None
                for attr in ["mlp", "feed_forward", "ffn"]:
                    if hasattr(layer, attr):
                        ffn_module = getattr(layer, attr)
                        break

                if ffn_module is not None:
                    def make_ffn_hook(idx):
                        def hook(module, args, output):
                            if isinstance(output, tuple):
                                ffn_out_storage[idx] = output[0].detach()
                            else:
                                ffn_out_storage[idx] = output.detach()
                        return hook

                    hooks.append(ffn_module.register_forward_hook(make_ffn_hook(layer_idx)))

            # Forward pass on extended prefix
            try:
                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                        use_cache=False,
                    )

                hidden_states = outputs.hidden_states  # (num_layers+1) tuple
                final_logits = outputs.logits

                # Determine greedy_token_id at decision position
                if greedy_token_id is None:
                    greedy_token_id = final_logits[0, -1, :].argmax().item()

                greedy_token = tokenizer.decode([greedy_token_id])

                # Process each layer at the decision position (last of extended prefix)
                for layer_idx in range(num_layers):
                    # hidden_states[layer_idx+1] is output of layer_idx
                    # Use decision_position (last position in extended prefix)
                    hidden = hidden_states[layer_idx + 1][:, -1, :]

                    # Apply final norm if available
                    if final_norm is not None:
                        normed = final_norm(hidden)
                    else:
                        normed = hidden

                    # Get logits
                    logits = lm_head(normed)
                    probs = torch.softmax(logits, dim=-1)

                    # p_sem_first_token (using context-aware metric_target_ids)
                    if metric_target_ids:
                        p_sem_first = probs[0, list(metric_target_ids)].sum().item()
                    else:
                        p_sem_first = 0.0

                    greedy_prob = probs[0, greedy_token_id].item()

                    logit_lens_rows.append({
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "decision_step": decision_step,
                        "word_present": word_present,
                        "metric_target_kind": metric_target_kind,
                        "layer": layer_idx,
                        "p_sem_first_token": p_sem_first,
                        "greedy_token": greedy_token,
                        "greedy_token_prob": greedy_prob,
                    })

                    # Decomposition at decision position
                    h_in = h_in_storage.get(layer_idx)
                    attn_out = attn_out_storage.get(layer_idx)
                    ffn_out = ffn_out_storage.get(layer_idx)

                    if h_in is not None and attn_out is not None and ffn_out is not None:
                        # Extract decision position (last position)
                        h = h_in[:, -1, :]
                        a = attn_out[:, -1, :]
                        f = ffn_out[:, -1, :]

                        def compute_p(state):
                            if final_norm is not None:
                                n = final_norm(state)
                            else:
                                n = state
                            lg = lm_head(n)
                            pr = torch.softmax(lg, dim=-1)
                            if metric_target_ids:
                                return pr[0, list(metric_target_ids)].sum().item()
                            return 0.0

                        p_h_in = compute_p(h)
                        p_h_plus_attn = compute_p(h + a)
                        p_h_out = compute_p(h + a + f)

                        # Fix D: Positive contribution means the module INCREASED target probability
                        # attn adds to h_in, so attn_contrib = p_after_attn - p_before_attn
                        # ffn adds to h+a, so ffn_contrib = p_after_ffn - p_before_ffn
                        attn_contrib = p_h_plus_attn - p_h_in
                        ffn_contrib = p_h_out - p_h_plus_attn

                        decomp_rows.append({
                            "prompt_id": prompt_id,
                            "condition": condition,
                            "decision_step": decision_step,
                            "word_present": word_present,
                            "metric_target_kind": metric_target_kind,
                            "layer": layer_idx,
                            "p_h_in": p_h_in,
                            "p_h_in_plus_attn": p_h_plus_attn,
                            "p_h_out": p_h_out,
                            "attn_contrib": attn_contrib,
                            "ffn_contrib": ffn_contrib,
                        })

            finally:
                # Remove hooks
                for hook in hooks:
                    hook.remove()

                # Clear storage
                h_in_storage.clear()
                attn_out_storage.clear()
                ffn_out_storage.clear()

                # Clear cache
                torch.cuda.empty_cache()

    if skipped_mapping_error > 0:
        logger.info("Skipped %d completions due to mapping_error", skipped_mapping_error)
    if skipped_no_entry > 0:
        logger.info("Skipped %d completions with missing detection entries", skipped_no_entry)
    if skipped_missing_trace > 0:
        logger.info("Skipped %d completions due to missing trace data", skipped_missing_trace)

    # Write outputs
    runs_dir.mkdir(parents=True, exist_ok=True)

    logit_lens_path = runs_dir / "logit_lens.csv"
    df_lens = pd.DataFrame(logit_lens_rows)
    df_lens.to_csv(logit_lens_path, index=False)
    logger.info("Wrote %d logit lens rows to %s", len(df_lens), logit_lens_path)

    decomp_path = runs_dir / "ffn_attn_decomp.csv"
    df_decomp = pd.DataFrame(decomp_rows)
    df_decomp.to_csv(decomp_path, index=False)
    logger.info("Wrote %d decomp rows to %s", len(df_decomp), decomp_path)

    return {
        "logit_lens_path": logit_lens_path,
        "ffn_attn_decomp_path": decomp_path,
    }


# ============================================================================
# PIPELINE ENTRYPOINT
# ============================================================================


def run_mechanistic_metrics_pipeline(
    output_root: Optional[Path] = None,
    prompts_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Run full mechanistic metrics pipeline.

    Steps:
    1. Compute attention metrics from traces
    2. Compute logit lens and attn/FFN decomposition

    Args:
        output_root: Run root directory
        prompts_path: Path to prompts.csv
        limit: Optional limit on prompts

    Returns:
        Dict with all output paths
    """
    results = {}

    logger.info("Step 1: Computing attention metrics...")
    try:
        attn_path = compute_attention_metrics(output_root, prompts_path, limit)
        results["attention_metrics_path"] = attn_path
    except Exception as e:
        logger.error("Attention metrics failed: %s", e)
        results["attention_metrics_error"] = str(e)

    logger.info("Step 2: Computing logit lens and decomposition...")
    try:
        lens_result = compute_logit_lens_and_decomp(output_root, prompts_path, limit)
        results.update(lens_result)
    except Exception as e:
        logger.error("Logit lens/decomp failed: %s", e)
        results["logit_lens_error"] = str(e)

    logger.info("Mechanistic metrics pipeline complete.")
    return results


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("METRICS_ATTN MODULE SELF-TEST")
    print("=" * 60)

    # Test 1: _compute_char_spans for baseline
    print("\n1. Testing _compute_char_spans (baseline):")
    baseline_prompt = (
        "Answer with exactly one English word.\n"
        "Question: What is 2+2?\n"
        "Answer:"
    )
    spans = _compute_char_spans(baseline_prompt, "four", "baseline")
    assert spans["instr_span"] is not None, "instr_span should not be None"
    assert spans["negation_span"] is None, "negation_span should be None for baseline"
    assert spans["question_span"] is not None, "question_span should not be None"
    instr_text = baseline_prompt[spans["instr_span"][0]:spans["instr_span"][1]]
    assert instr_text == "Answer with exactly one English word.", f"Got: {instr_text!r}"
    print("   PASS: Baseline spans computed correctly")

    # Test 2: _compute_char_spans for negative
    print("\n2. Testing _compute_char_spans (negative):")
    negative_prompt = (
        "Answer with exactly one English word.\n"
        'Do not use the word "four" anywhere in your answer.\n'
        "Question: What is 2+2?\n"
        "Answer:"
    )
    spans = _compute_char_spans(negative_prompt, "four", "negative")
    assert spans["instr_span"] is not None, "instr_span should not be None"
    assert spans["negation_span"] is not None, "negation_span should not be None"
    assert spans["target_mention_span"] is not None, "target_mention_span should not be None"
    negation_text = negative_prompt[spans["negation_span"][0]:spans["negation_span"][1]]
    assert negation_text == "Do not use", f"Got: {negation_text!r}"
    target_text = negative_prompt[spans["target_mention_span"][0]:spans["target_mention_span"][1]]
    assert target_text == "four", f"Got: {target_text!r}"
    print("   PASS: Negative spans computed correctly")

    # Test 3: _map_char_span_to_token_indices
    print("\n3. Testing _map_char_span_to_token_indices:")
    offsets = [(0, 5), (5, 10), (10, 15), (15, 20)]
    indices = _map_char_span_to_token_indices(offsets, (7, 12))
    assert indices == [1, 2], f"Expected [1, 2], got {indices}"
    indices_none = _map_char_span_to_token_indices(offsets, None)
    assert indices_none == [], f"Expected [], got {indices_none}"
    print("   PASS: Token index mapping works")

    # Test 4: _resolve_run_root with None
    print("\n4. Testing _resolve_run_root:")
    try:
        import warnings as w
        with w.catch_warnings(record=True):
            w.simplefilter("always")
            result = _resolve_run_root(None)
            print(f"   Resolved to: {result}")
            print("   PASS: _resolve_run_root handles None")
    except Exception as e:
        print(f"   SKIP: {e}")

    print("\n" + "=" * 60)
    print("All self-tests passed!")
    print("=" * 60)
