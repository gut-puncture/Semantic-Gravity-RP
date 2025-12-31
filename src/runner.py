"""
runner.py - Main Model Runner for Semantic Gravity Experiment

This module provides:
- Greedy (mechanistic) runs with full hidden state/attention capture
- Stochastic (behavioral) sampling runs
- Resumable execution with checkpoint detection
- Memory management for GPU efficiency

Per specification Section 8 and execution-plan Section 7.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS
# ============================================================================

try:
    from .utils import ModelWrapper
    from .prompt_builder import build_prompt
    from .config import setup_directories, get_base_paths
except ImportError:
    from utils import ModelWrapper
    from prompt_builder import build_prompt
    from config import setup_directories, get_base_paths


# ============================================================================
# COMPLETION TRACKING
# ============================================================================


def load_existing_completions(jsonl_path: Path) -> Set[str]:
    """
    Load existing completion keys from JSONL file.

    Returns set of "prompt_id|condition|sample_id" keys.
    """
    existing = set()
    if not jsonl_path.exists():
        return existing

    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    prompt_id = record.get("prompt_id", "")
                    condition = record.get("condition", "")
                    sample_id = record.get("sample_id", "")
                    key = f"{prompt_id}|{condition}|{sample_id}"
                    existing.add(key)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Could not load existing completions from {jsonl_path}: {e}")

    return existing


def append_completion(jsonl_path: Path, record: Dict) -> None:
    """Append a completion record to JSONL file."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================================
# BATCH HELPERS
# ============================================================================


def _iter_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _trim_generated_ids(
    token_ids: List[int],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> List[int]:
    """Trim trailing padding tokens; keep a single EOS if present."""
    if eos_token_id is not None and eos_token_id in token_ids:
        eos_idx = token_ids.index(eos_token_id)
        return token_ids[:eos_idx + 1]
    if pad_token_id is not None:
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids = token_ids[:-1]
    return token_ids


def _tokenize_batch(wrapper: ModelWrapper, prompt_texts: List[str]) -> Dict[str, Any]:
    from .config import CONFIG

    add_special_tokens = CONFIG.get("model", {}).get("add_special_tokens", False)
    tokenizer = wrapper.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=add_special_tokens,
    )
    input_ids = enc["input_ids"].to(wrapper.model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        attention_mask = (input_ids != pad_token_id).long()
    attention_mask = attention_mask.to(wrapper.model.device)
    input_lengths = attention_mask.sum(dim=1).tolist()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_lengths": input_lengths,
    }

# ============================================================================
# MECHANISTIC (GREEDY) RUNNER
# ============================================================================


def run_mechanistic(
    wrapper: ModelWrapper,
    prompt_text: str,
    prompt_id: str,
    condition: str,
    output_dir: Path,
    max_new_tokens: int = 8,
) -> Optional[Dict]:
    """
    Run greedy mechanistic inference with full state capture.

    Args:
        wrapper: Loaded ModelWrapper instance
        prompt_text: Full prompt string
        prompt_id: Prompt identifier
        condition: "baseline" or "negative"
        output_dir: Directory for trace files
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with generated_text, generated_ids, or None if skipped
    """
    import torch

    trace_path = output_dir / f"{prompt_id}_{condition}.pt"

    # Skip if trace already exists
    if trace_path.exists():
        logger.debug(f"Skipping {prompt_id}_{condition} - trace exists")
        return None

    # Tokenize input
    inputs = wrapper.tokenize(prompt_text)
    input_ids = inputs["input_ids"]
    input_length = input_ids.shape[1]

    # Generate with full state capture
    with torch.no_grad():
        outputs = wrapper.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=wrapper.tokenizer.pad_token_id,
        )

    # Extract generated portion
    generated_ids = outputs.sequences[:, input_length:]
    generated_ids_list = generated_ids[0].tolist()
    generated_ids_list = _trim_generated_ids(
        generated_ids_list,
        wrapper.tokenizer.eos_token_id,
        wrapper.tokenizer.pad_token_id,
    )
    generated_text = wrapper.tokenizer.decode(
        generated_ids_list,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Prepare trace data (move to CPU)
    trace_data = {
        "prompt_id": prompt_id,
        "condition": condition,
        "input_ids": input_ids.cpu(),
        "generated_ids": generated_ids.cpu(),
    }

    # Handle hidden states (list of tuples)
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        # hidden_states is tuple of (num_generated_tokens) tuples
        # Each inner tuple has (num_layers) tensors
        hidden_states_cpu = []
        for step_states in outputs.hidden_states:
            step_cpu = tuple(h.cpu() for h in step_states)
            hidden_states_cpu.append(step_cpu)
        trace_data["hidden_states"] = hidden_states_cpu

    # Handle attentions (list of tuples)
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        attentions_cpu = []
        for step_attn in outputs.attentions:
            step_cpu = tuple(a.cpu() for a in step_attn)
            attentions_cpu.append(step_cpu)
        trace_data["attentions"] = attentions_cpu

    # Save trace
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trace_data, trace_path)
    logger.debug(f"Saved trace to {trace_path}")

    # Cleanup
    del outputs
    del inputs
    torch.cuda.empty_cache()

    # Determine finish reason
    finish_reason = "length"  # Default for greedy
    if len(generated_ids_list) < max_new_tokens:
        finish_reason = "stop"

    return {
        "generated_text": generated_text,
        "generated_ids": generated_ids_list,
        "finish_reason": finish_reason,
    }


# ============================================================================
# BEHAVIORAL (SAMPLING) RUNNER
# ============================================================================


def run_behavioral(
    wrapper: ModelWrapper,
    prompt_text: str,
    prompt_id: str,
    condition: str,
    existing_keys: Set[str],
    completions_path: Path,
    num_samples: int = 16,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> List[Dict]:
    """
    Run stochastic sampling for behavioral analysis.

    Args:
        wrapper: Loaded ModelWrapper instance
        prompt_text: Full prompt string
        prompt_id: Prompt identifier
        condition: "baseline" or "negative"
        existing_keys: Set of already-completed sample keys
        completions_path: Path to completions JSONL
        num_samples: Number of samples to generate
        max_new_tokens: Max tokens per sample
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        List of completion records
    """
    import torch

    results = []

    # Check which samples already exist
    samples_to_run = []
    for sample_id in range(num_samples):
        key = f"{prompt_id}|{condition}|{sample_id}"
        if key not in existing_keys:
            samples_to_run.append(sample_id)

    if not samples_to_run:
        logger.debug(f"Skipping {prompt_id}_{condition} - all samples exist")
        return results

    # Tokenize input
    inputs = wrapper.tokenize(prompt_text)
    input_length = inputs["input_ids"].shape[1]

    # Generate all samples at once
    with torch.no_grad():
        outputs = wrapper.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=len(samples_to_run),
            pad_token_id=wrapper.tokenizer.pad_token_id,
        )

    # Extract and save each sample
    timestamp = time.time()
    for i, sample_id in enumerate(samples_to_run):
        generated_ids = outputs[i, input_length:]
        generated_ids_list = generated_ids.tolist()
        generated_ids_list = _trim_generated_ids(
            generated_ids_list,
            wrapper.tokenizer.eos_token_id,
            wrapper.tokenizer.pad_token_id,
        )
        generated_text = wrapper.tokenizer.decode(
            generated_ids_list,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Determine finish reason
        finish_reason = "length"
        if len(generated_ids_list) < max_new_tokens:
            finish_reason = "stop"

        record = {
            "prompt_id": prompt_id,
            "condition": condition,
            "sample_id": sample_id,
            "generated_text": generated_text,
            "generated_token_ids": generated_ids_list,
            "finish_reason": finish_reason,
            "timestamp": timestamp,
        }

        append_completion(completions_path, record)
        results.append(record)

    # Cleanup
    del outputs
    del inputs
    torch.cuda.empty_cache()

    return results


# ============================================================================
# MAIN RUNNER
# ============================================================================


def run_experiment(
    prompts_csv: Optional[str] = None,
    output_root: Optional[str] = None,
    skip_mechanistic: bool = False,
    skip_behavioral: bool = False,
    limit: Optional[int] = None,
    mechanistic_batch_size: int = 1,
    behavioral_batch_size: int = 4,
    log_every: int = 50,
) -> Dict[str, int]:
    """
    Run the full experiment on all prompts.

    Args:
        prompts_csv: Path to prompts CSV (defaults to data/prompts.csv)
        output_root: Root output directory (defaults to setup_directories)
        skip_mechanistic: Skip greedy/mechanistic runs
        skip_behavioral: Skip sampling runs
        limit: Limit number of prompts to process (for testing)
        mechanistic_batch_size: Batch size for mechanistic runs (same-length prompts only)
        behavioral_batch_size: Batch size for behavioral runs
        log_every: Log progress every N prompts or samples

    Returns:
        Dict with counts of completed runs
    """
    import pandas as pd
    import torch
    from collections import defaultdict
    from .config import CONFIG

    # Setup paths
    if output_root is None:
        dirs = setup_directories()
        output_root = dirs["run_root"]
    else:
        output_root = Path(output_root)

    if prompts_csv is None:
        base_paths = get_base_paths()
        prompts_csv = base_paths["data_root"] / "prompts.csv"
    else:
        prompts_csv = Path(prompts_csv)

    # Output paths
    mechanistic_dir = output_root / "runs" / "mechanistic_trace"
    greedy_jsonl = output_root / "runs" / "completions_greedy.jsonl"
    samples_jsonl = output_root / "runs" / "completions_samples.jsonl"

    mechanistic_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    logger.info(f"Loading prompts from {prompts_csv}")
    df = pd.read_csv(prompts_csv)

    # Require question_text explicitly
    required_cols = ["prompt_id", "question_text", "target_word", "category"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Ensure prompts.csv includes 'question_text' (raw cloze question)."
        )

    if limit:
        df = df.head(limit)

    logger.info(f"Processing {len(df)} prompts")

    # Load ModelWrapper
    wrapper = ModelWrapper.get_instance()
    if not wrapper.is_loaded:
        logger.info("Loading model...")
        wrapper.load()

    # Load existing completions for resumability
    existing_greedy = load_existing_completions(greedy_jsonl)
    existing_samples = load_existing_completions(samples_jsonl)

    # Counters
    counts = {
        "mechanistic_baseline": 0,
        "mechanistic_negative": 0,
        "greedy_baseline": 0,
        "greedy_negative": 0,
        "samples_baseline": 0,
        "samples_negative": 0,
    }

    model_cfg = CONFIG.get("model", {})
    max_new_tokens_greedy = int(model_cfg.get("max_new_tokens_greedy", 8))
    max_new_tokens_stochastic = int(model_cfg.get("max_new_tokens_stochastic", 10))
    num_samples = int(model_cfg.get("num_stochastic_samples", 16))
    temperature = float(model_cfg.get("temperature", 1.0))
    top_p = float(model_cfg.get("top_p", 0.9))
    add_special_tokens = model_cfg.get("add_special_tokens", False)

    # Build task lists
    mechanistic_tasks: List[Dict[str, Any]] = []
    behavioral_tasks: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        prompt_id = str(row["prompt_id"])
        question_text = row["question_text"]
        target_word = row["target_word"]

        for condition in ["baseline", "negative"]:
            prompt_text = build_prompt(question_text, target_word, condition)
            greedy_key = f"{prompt_id}|{condition}|"
            trace_path = mechanistic_dir / f"{prompt_id}_{condition}.pt"

            if not skip_mechanistic and not trace_path.exists():
                input_len = len(
                    wrapper.tokenizer.encode(prompt_text, add_special_tokens=add_special_tokens)
                )
                mechanistic_tasks.append({
                    "prompt_id": prompt_id,
                    "condition": condition,
                    "prompt_text": prompt_text,
                    "trace_path": trace_path,
                    "greedy_key": greedy_key,
                    "input_len": input_len,
                })

            if not skip_behavioral:
                samples_to_run = []
                for sample_id in range(num_samples):
                    key = f"{prompt_id}|{condition}|{sample_id}"
                    if key not in existing_samples:
                        samples_to_run.append(sample_id)
                if samples_to_run:
                    behavioral_tasks.append({
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "prompt_text": prompt_text,
                        "samples_to_run": samples_to_run,
                    })

    # Mechanistic runs (grouped by input length to avoid padding artifacts)
    if not skip_mechanistic and mechanistic_tasks:
        length_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for task in mechanistic_tasks:
            length_groups[task["input_len"]].append(task)

        logger.info(
            "Mechanistic tasks: %d across %d length groups (batch=%d)",
            len(mechanistic_tasks),
            len(length_groups),
            max(1, int(mechanistic_batch_size)),
        )

        processed = 0
        last_log = 0
        start_time = time.monotonic()

        def _process_mechanistic_batch(batch: List[Dict[str, Any]]) -> None:
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
                    max_new_tokens=max_new_tokens_greedy,
                    do_sample=False,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    pad_token_id=wrapper.tokenizer.pad_token_id,
                )

            if getattr(outputs, "attentions", None) is None:
                raise RuntimeError(
                    "Model returned no attentions. Ensure attention implementation is set to eager."
                )
            if any(step is None for step in outputs.attentions):
                raise RuntimeError(
                    "Model returned None attention steps. Ensure attention implementation is set to eager."
                )

            sequences = outputs.sequences
            eos_token_id = wrapper.tokenizer.eos_token_id
            pad_token_id = wrapper.tokenizer.pad_token_id

            for i, task in enumerate(batch):
                prompt_id = task["prompt_id"]
                condition = task["condition"]
                trace_path = task["trace_path"]
                greedy_key = task["greedy_key"]

                generated_ids = sequences[i, batch_input_len:].tolist()
                generated_ids = _trim_generated_ids(generated_ids, eos_token_id, pad_token_id)
                gen_len = len(generated_ids)
                generated_text = wrapper.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                trace_data = {
                    "prompt_id": prompt_id,
                    "condition": condition,
                    "input_ids": input_ids[i:i + 1, :batch_input_len].cpu(),
                    "generated_ids": torch.tensor([generated_ids], dtype=torch.long),
                }

                if getattr(outputs, "hidden_states", None):
                    hidden_states_cpu = []
                    for step_states in outputs.hidden_states[:gen_len]:
                        step_cpu = tuple(h[i:i + 1].cpu() for h in step_states)
                        hidden_states_cpu.append(step_cpu)
                    trace_data["hidden_states"] = hidden_states_cpu

                if getattr(outputs, "attentions", None):
                    attentions_cpu = []
                    for step_attn in outputs.attentions[:gen_len]:
                        step_cpu = tuple(a[i:i + 1].cpu() for a in step_attn)
                        attentions_cpu.append(step_cpu)
                    trace_data["attentions"] = attentions_cpu

                torch.save(trace_data, trace_path)
                counts[f"mechanistic_{condition}"] += 1

                if greedy_key not in existing_greedy:
                    greedy_record = {
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "sample_id": "",
                        "prompt_text": task["prompt_text"],
                        "generated_text": generated_text,
                        "generated_token_ids": generated_ids,
                        "finish_reason": "stop" if gen_len < max_new_tokens_greedy else "length",
                        "timestamp": time.time(),
                    }
                    append_completion(greedy_jsonl, greedy_record)
                    existing_greedy.add(greedy_key)
                    counts[f"greedy_{condition}"] += 1

            del outputs
            del input_ids
            del attention_mask
            torch.cuda.empty_cache()

            processed += len(batch)
            if log_every and (processed - last_log) >= log_every:
                elapsed = max(time.monotonic() - start_time, 1e-6)
                rate = processed / elapsed
                remaining = len(mechanistic_tasks) - processed
                eta = remaining / rate if rate > 0 else 0.0
                logger.info(
                    "Mechanistic progress: %d/%d (%.1f%%) | %.2f prompts/s | ETA %.1fs",
                    processed,
                    len(mechanistic_tasks),
                    100.0 * processed / len(mechanistic_tasks),
                    rate,
                    eta,
                )
                last_log = processed

        def _run_mechanistic_batch(batch: List[Dict[str, Any]]) -> None:
            try:
                _process_mechanistic_batch(batch)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                logger.warning("OOM in mechanistic batch of %d; splitting", len(batch))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(batch) <= 1:
                    raise
                mid = len(batch) // 2
                _run_mechanistic_batch(batch[:mid])
                _run_mechanistic_batch(batch[mid:])

        mech_batch_size = max(1, int(mechanistic_batch_size))
        for _, tasks in length_groups.items():
            for batch in _iter_batches(tasks, mech_batch_size):
                _run_mechanistic_batch(batch)

    # Recovery: if trace exists but greedy JSONL missing, recover from trace
    if not skip_mechanistic:
        for _, row in df.iterrows():
            prompt_id = str(row["prompt_id"])
            question_text = row["question_text"]
            target_word = row["target_word"]
            for condition in ["baseline", "negative"]:
                prompt_text = build_prompt(question_text, target_word, condition)
                greedy_key = f"{prompt_id}|{condition}|"
                trace_path = mechanistic_dir / f"{prompt_id}_{condition}.pt"
                if greedy_key in existing_greedy or not trace_path.exists():
                    continue
                try:
                    trace_data = torch.load(trace_path, map_location="cpu")
                    generated_ids = trace_data["generated_ids"]
                    if hasattr(generated_ids, "tolist"):
                        generated_ids_list = generated_ids[0].tolist() if generated_ids.dim() > 1 else generated_ids.tolist()
                    else:
                        generated_ids_list = list(generated_ids)
                    generated_text = wrapper.tokenizer.decode(
                        generated_ids_list,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    greedy_record = {
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "sample_id": "",
                        "prompt_text": prompt_text,
                        "generated_text": generated_text,
                        "generated_token_ids": generated_ids_list,
                        "finish_reason": "unknown",
                        "timestamp": time.time(),
                    }
                    append_completion(greedy_jsonl, greedy_record)
                    existing_greedy.add(greedy_key)
                    counts[f"greedy_{condition}"] += 1
                    logger.debug("Recovered greedy completion from trace: %s_%s", prompt_id, condition)
                except Exception as e:
                    logger.warning("Failed to recover greedy from trace %s: %s", trace_path, e)

    # Behavioral runs (group by missing sample count)
    if not skip_behavioral and behavioral_tasks:
        sample_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for task in behavioral_tasks:
            sample_groups[len(task["samples_to_run"])].append(task)

        total_behavioral_samples = sum(len(t["samples_to_run"]) for t in behavioral_tasks)

        logger.info(
            "Behavioral tasks: %d prompts across %d groups (batch=%d)",
            len(behavioral_tasks),
            len(sample_groups),
            max(1, int(behavioral_batch_size)),
        )

        processed_samples = 0
        last_log = 0
        start_time = time.monotonic()

        def _process_behavioral_batch(batch: List[Dict[str, Any]], num_to_generate: int) -> None:
            nonlocal processed_samples, last_log

            prompt_texts = [t["prompt_text"] for t in batch]
            tokenized = _tokenize_batch(wrapper, prompt_texts)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            batch_input_len = input_ids.shape[1]

            with torch.inference_mode():
                outputs = wrapper.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens_stochastic,
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
                condition = task["condition"]
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
                    finish_reason = "stop" if len(generated_ids) < max_new_tokens_stochastic else "length"
                    record = {
                        "prompt_id": prompt_id,
                        "condition": condition,
                        "sample_id": sample_id,
                        "generated_text": generated_text,
                        "generated_token_ids": generated_ids,
                        "finish_reason": finish_reason,
                        "timestamp": timestamp,
                    }
                    append_completion(samples_jsonl, record)
                    key = f"{prompt_id}|{condition}|{sample_id}"
                    existing_samples.add(key)
                    counts[f"samples_{condition}"] += 1
                    processed_samples += 1

            del outputs
            del input_ids
            del attention_mask
            torch.cuda.empty_cache()

            if log_every and (processed_samples - last_log) >= log_every:
                elapsed = max(time.monotonic() - start_time, 1e-6)
                rate = processed_samples / elapsed
                remaining = total_behavioral_samples - processed_samples
                eta = remaining / rate if rate > 0 else 0.0
                logger.info(
                    "Behavioral progress: %d/%d samples (%.1f%%) | %.2f samples/s | ETA %.1fs",
                    processed_samples,
                    total_behavioral_samples,
                    100.0 * processed_samples / max(1, total_behavioral_samples),
                    rate,
                    eta,
                )
                last_log = processed_samples

        def _run_behavioral_batch(batch: List[Dict[str, Any]], num_to_generate: int) -> None:
            try:
                _process_behavioral_batch(batch, num_to_generate)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                logger.warning("OOM in behavioral batch of %d; splitting", len(batch))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(batch) <= 1:
                    raise
                mid = len(batch) // 2
                _run_behavioral_batch(batch[:mid], num_to_generate)
                _run_behavioral_batch(batch[mid:], num_to_generate)

        beh_batch_size = max(1, int(behavioral_batch_size))
        for num_to_generate, tasks in sample_groups.items():
            if num_to_generate <= 0:
                continue
            for batch in _iter_batches(tasks, beh_batch_size):
                _run_behavioral_batch(batch, num_to_generate)

    counts["mechanistic_completed"] = counts["mechanistic_baseline"] + counts["mechanistic_negative"]
    counts["greedy_completed"] = counts["greedy_baseline"] + counts["greedy_negative"]
    counts["behavioral_completed"] = counts["samples_baseline"] + counts["samples_negative"]

    logger.info(f"Experiment complete. Counts: {counts}")
    return counts


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("RUNNER MODULE SELF-TEST")
    print("=" * 60)

    # Test imports
    print("\n1. Testing imports:")
    try:
        from utils import ModelWrapper
        from prompt_builder import build_prompt
        print("   PASS: All imports successful")
    except ImportError as e:
        print(f"   FAIL: Import error: {e}")
        sys.exit(1)

    # Test build_prompt
    print("\n2. Testing build_prompt integration:")
    baseline = build_prompt("What is 2+2?", "four", "baseline")
    negative = build_prompt("What is 2+2?", "four", "negative")
    assert "Answer with exactly one English word" in baseline
    assert "Do not use" in negative
    print("   PASS: build_prompt works correctly")

    # Test completion tracking
    print("\n3. Testing completion tracking:")
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_jsonl = Path(tmpdir) / "test.jsonl"

        # Write a test record
        record = {"prompt_id": "test1", "condition": "baseline", "sample_id": 0}
        append_completion(test_jsonl, record)

        # Load and verify
        existing = load_existing_completions(test_jsonl)
        assert "test1|baseline|0" in existing
        print("   PASS: Completion tracking works")

    print("\n" + "=" * 60)
    print("Runner module self-test complete!")
    print("=" * 60)
    print("\nNote: Full experiment requires prompts.csv and GPU.")
