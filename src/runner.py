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
    generated_text = wrapper.tokenizer.decode(
        generated_ids[0],
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
    if generated_ids.shape[1] < max_new_tokens:
        finish_reason = "stop"

    return {
        "generated_text": generated_text,
        "generated_ids": generated_ids[0].tolist(),
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
        generated_text = wrapper.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Determine finish reason
        finish_reason = "length"
        if len(generated_ids) < max_new_tokens:
            finish_reason = "stop"

        record = {
            "prompt_id": prompt_id,
            "condition": condition,
            "sample_id": sample_id,
            "generated_text": generated_text,
            "generated_token_ids": generated_ids.tolist(),
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
) -> Dict[str, int]:
    """
    Run the full experiment on all prompts.

    Args:
        prompts_csv: Path to prompts CSV (defaults to data/prompts.csv)
        output_root: Root output directory (defaults to setup_directories)
        skip_mechanistic: Skip greedy/mechanistic runs
        skip_behavioral: Skip sampling runs
        limit: Limit number of prompts to process (for testing)

    Returns:
        Dict with counts of completed runs
    """
    import pandas as pd
    import torch

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

    # Process each prompt
    for idx, row in df.iterrows():
        prompt_id = str(row["prompt_id"])
        question_text = row["question_text"]  # Required column
        target_word = row["target_word"]
        category = row["category"]

        logger.info(f"Processing prompt {idx + 1}/{len(df)}: {prompt_id}")

        for condition in ["baseline", "negative"]:
            # Build prompt
            prompt_text = build_prompt(question_text, target_word, condition)
            greedy_key = f"{prompt_id}|{condition}|"
            trace_path = mechanistic_dir / f"{prompt_id}_{condition}.pt"

            # Mechanistic run
            if not skip_mechanistic:
                result = run_mechanistic(
                    wrapper=wrapper,
                    prompt_text=prompt_text,
                    prompt_id=prompt_id,
                    condition=condition,
                    output_dir=mechanistic_dir,
                )
                if result:
                    counts[f"mechanistic_{condition}"] += 1

                    # Also save greedy completion to JSONL
                    if greedy_key not in existing_greedy:
                        greedy_record = {
                            "prompt_id": prompt_id,
                            "condition": condition,
                            "sample_id": "",
                            "prompt_text": prompt_text,
                            "generated_text": result["generated_text"],
                            "generated_token_ids": result["generated_ids"],
                            "finish_reason": result.get("finish_reason", "unknown"),
                            "timestamp": time.time(),
                        }
                        append_completion(greedy_jsonl, greedy_record)
                        existing_greedy.add(greedy_key)
                        counts[f"greedy_{condition}"] += 1

            # Recovery: if trace exists but greedy JSONL missing, recover from trace
            if greedy_key not in existing_greedy and trace_path.exists():
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
                    logger.debug(f"Recovered greedy completion from trace: {prompt_id}_{condition}")
                except Exception as e:
                    logger.warning(f"Failed to recover greedy from trace {trace_path}: {e}")

            # Behavioral sampling
            if not skip_behavioral:
                samples = run_behavioral(
                    wrapper=wrapper,
                    prompt_text=prompt_text,
                    prompt_id=prompt_id,
                    condition=condition,
                    existing_keys=existing_samples,
                    completions_path=samples_jsonl,
                )
                counts[f"samples_{condition}"] += len(samples)

                # Update existing keys
                for rec in samples:
                    key = f"{rec['prompt_id']}|{rec['condition']}|{rec['sample_id']}"
                    existing_samples.add(key)

        # Memory cleanup after each prompt
        torch.cuda.empty_cache()

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
