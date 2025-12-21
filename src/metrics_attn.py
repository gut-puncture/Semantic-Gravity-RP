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
    from .utils import ModelWrapper, compute_token_char_spans
    from .prompt_builder import build_prompt
    from .config import get_base_paths, CONFIG
    from .metrics_psem import token_sequences_for_variants
except ImportError:
    from utils import ModelWrapper, compute_token_char_spans
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
            warnings.warn(
                f"Selecting latest experiment_run_* under {candidate}: {latest}",
                UserWarning,
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
        warnings.warn(f"output_root not provided; using latest run dir: {latest}", UserWarning)
        return latest

    # Fallback to base_out
    return base_out


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

    Tries tokenizer's return_offsets_mapping first, falls back to
    incremental decode if lengths mismatch.

    Args:
        prompt_text: The prompt text string
        tokenizer: Tokenizer instance
        input_ids: List of token IDs

    Returns:
        List of (start, end) character offsets for each token
    """
    # Try with add_special_tokens=True
    try:
        encoded = tokenizer(
            prompt_text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping", [])
        if len(offsets) == len(input_ids):
            return offsets
    except Exception:
        pass

    # Try with add_special_tokens=False
    try:
        encoded = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping", [])
        if len(offsets) == len(input_ids):
            return offsets
    except Exception:
        pass

    # Fallback to incremental decode
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

    Metrics computed per layer/head for the first generated token:
    - IAR: Instruction Attention Ratio
    - NF: Negation Focus (within instruction span)
    - TMF: Target Mention Focus (within instruction span)
    - PI: Polarity Index (TMF - NF)

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

    # Get tokenizer
    wrapper = ModelWrapper.get_instance()
    if wrapper.tokenizer is None:
        wrapper.load()
    tokenizer = wrapper.tokenizer

    rows = []

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

            # Build prompt and get spans
            prompt_text = build_prompt(question_text, target_word, condition)
            char_spans = _compute_char_spans(prompt_text, target_word, condition)

            # Get offsets and map to token indices
            offsets = _get_offsets_for_prompt(prompt_text, tokenizer, input_ids_list)

            instr_tokens = _map_char_span_to_token_indices(offsets, char_spans["instr_span"])
            negation_tokens = _map_char_span_to_token_indices(offsets, char_spans["negation_span"])
            target_tokens = _map_char_span_to_token_indices(offsets, char_spans["target_mention_span"])
            question_tokens = _map_char_span_to_token_indices(offsets, char_spans["question_span"])

            # Step 0 attentions (first generated token)
            step0_attn = attentions[0]  # Tuple of layer tensors

            num_layers = len(step0_attn)
            all_head_metrics = []

            for layer_idx, layer_attn in enumerate(step0_attn):
                # layer_attn shape: [1, num_heads, seq_len, seq_len]
                # For generated token, we look at last position attending to context
                if layer_attn.dim() < 4:
                    continue

                num_heads = layer_attn.shape[1]
                seq_len = layer_attn.shape[2]
                gen_pos = seq_len - 1  # Last position = generated token

                for head_idx in range(num_heads):
                    # Attention weights from gen_pos to all positions
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
                    "layer": -1,
                    "head": -1,
                    "iar": global_iar,
                    "nf": global_nf,
                    "tmf": global_tmf,
                    "pi": global_pi,
                    "aggregate_flag": "global_mean",
                })

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
    Compute logit lens and attention/FFN decomposition via forward pass.

    For each prompt/condition:
    - Runs forward pass with hooks to capture layer activations
    - Computes p_sem at each layer (logit lens)
    - Decomposes into attention and FFN contributions

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

    for _, row in prompts_df.iterrows():
        prompt_id = str(row["prompt_id"])
        question_text = row["question_text"]
        target_word = row["target_word"]

        # Get first token IDs for target
        seqs = token_sequences_for_variants(target_word, tokenizer)
        first_ids = sorted({seq[0] for seq in seqs if seq})

        for condition in ["baseline", "negative"]:
            prompt_text = build_prompt(question_text, target_word, condition)

            # Tokenize
            inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            # Try to get greedy_token_id from trace
            greedy_token_id = None
            trace_path = trace_dir / f"{prompt_id}_{condition}.pt"
            if trace_path.exists():
                try:
                    trace = torch.load(trace_path, map_location="cpu")
                    gen_ids = trace.get("generated_ids")
                    if gen_ids is not None:
                        if hasattr(gen_ids, "item"):
                            greedy_token_id = gen_ids[0][0].item() if gen_ids.dim() > 1 else gen_ids[0].item()
                        elif hasattr(gen_ids, "__getitem__"):
                            greedy_token_id = int(gen_ids[0][0]) if hasattr(gen_ids[0], "__getitem__") else int(gen_ids[0])
                except Exception:
                    pass

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

            # Forward pass
            try:
                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                        use_cache=False,
                    )

                hidden_states = outputs.hidden_states  # (num_layers+1) tuple
                final_logits = outputs.logits

                # Determine greedy_token_id from final logits if not from trace
                if greedy_token_id is None:
                    greedy_token_id = final_logits[0, -1, :].argmax().item()

                greedy_token = tokenizer.decode([greedy_token_id])

                # Process each layer
                for layer_idx in range(num_layers):
                    # hidden_states[layer_idx+1] is output of layer_idx
                    hidden = hidden_states[layer_idx + 1][:, -1, :]  # Last position

                    # Apply final norm if available
                    if final_norm is not None:
                        normed = final_norm(hidden)
                    else:
                        normed = hidden

                    # Get logits
                    logits = lm_head(normed)
                    probs = torch.softmax(logits, dim=-1)

                    # p_sem_first_token
                    if first_ids:
                        p_sem_first = probs[0, list(first_ids)].sum().item()
                    else:
                        p_sem_first = 0.0

                    greedy_prob = probs[0, greedy_token_id].item()

                    logit_lens_rows.append({
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "layer": layer_idx,
                        "p_sem_first_token": p_sem_first,
                        "greedy_token": greedy_token,
                        "greedy_token_prob": greedy_prob,
                    })

                    # Decomposition
                    h_in = h_in_storage.get(layer_idx)
                    attn_out = attn_out_storage.get(layer_idx)
                    ffn_out = ffn_out_storage.get(layer_idx)

                    if h_in is not None and attn_out is not None and ffn_out is not None:
                        # Extract last position
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
                            if first_ids:
                                return pr[0, list(first_ids)].sum().item()
                            return 0.0

                        p_h_in = compute_p(h)
                        p_h_plus_attn = compute_p(h + a)
                        p_h_out = compute_p(h + a + f)

                        attn_contrib = p_h_in - p_h_plus_attn
                        ffn_contrib = p_h_plus_attn - p_h_out

                        decomp_rows.append({
                            "prompt_id": prompt_id,
                            "condition": condition,
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
