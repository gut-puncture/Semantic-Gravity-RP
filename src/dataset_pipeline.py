"""
dataset_pipeline.py - End-to-end dataset construction for Module 2.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .api_clients import DeepSeekClient
from .config import CONFIG, get_base_paths
from .data_mining import CandidatePrompt, DatasetGenerator
from .utils import ModelWrapper
from .validator import (
    PromptSelector,
    PromptValidator,
    TargetTracker,
    ValidatedPrompt,
    validate_and_enrich_prompts,
)

logger = logging.getLogger(__name__)


def _compute_prompt_id(category: str, target_word_normalized: str, question_text: str, target_word: str) -> str:
    """
    Compute a unique prompt_id with content hash to prevent collisions.
    
    Format: {category}_{target_word_normalized}_{hash8}
    where hash8 = first 8 chars of SHA256(question_text||target_word||category)
    
    Args:
        category: The prompt category
        target_word_normalized: Normalized target word
        question_text: The question text
        target_word: The original target word
        
    Returns:
        Unique prompt_id string
    """
    import hashlib
    content = f"{question_text}||{target_word}||{category}"
    hash_full = hashlib.sha256(content.encode('utf-8')).hexdigest()
    hash8 = hash_full[:8]
    return f"{category}_{target_word_normalized}_{hash8}"


def _write_jsonl(items: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            json.dump(item, f)
            f.write('\n')


def save_candidates(
    candidates_by_category: Dict[str, List[CandidatePrompt]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_items: List[dict] = []

    for category, candidates in candidates_by_category.items():
        payload = [c.to_dict() for c in candidates]
        _write_jsonl(payload, output_dir / f"{category}.jsonl")
        all_items.extend(payload)

    _write_jsonl(all_items, output_dir / "all_candidates.jsonl")


def group_by_target(
    validated_prompts: List[ValidatedPrompt],
) -> Dict[str, List[ValidatedPrompt]]:
    groups: Dict[str, List[ValidatedPrompt]] = {}
    for vp in validated_prompts:
        key = vp.candidate.target_word_normalized
        groups.setdefault(key, []).append(vp)
    return groups


def build_dataset(
    prompts_per_category: int = CONFIG['dataset']['prompts_per_category'],
    candidate_multiplier: int = 2,
    candidates_per_target: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    deepseek_client: Optional[DeepSeekClient] = None,
    model_wrapper: Optional[ModelWrapper] = None,
    skip_generated: bool = False,
    use_fallback: bool = False,
    output_root: Optional[Path] = None,
) -> Dict[str, List[ValidatedPrompt]]:
    """
    Generate, validate, score, and select prompts for Module 2.
    """
    paths = get_base_paths()
    data_root = output_root or paths['data_root']
    cache_dir = cache_dir or (data_root / "raw")

    candidates_dir = data_root / "candidates"
    validated_dir = data_root / "validated"

    generator = DatasetGenerator(cache_dir=cache_dir, deepseek_client=deepseek_client)
    candidates_by_category = generator.generate_all(
        prompts_per_category=prompts_per_category,
        candidate_multiplier=candidate_multiplier,
        candidates_per_target=candidates_per_target,
        skip_generated=skip_generated,
        use_fallback=use_fallback,
    )

    save_candidates(candidates_by_category, candidates_dir)

    validator = PromptValidator(deepseek_client)
    tracker = TargetTracker(max_repetition=CONFIG['dataset']['max_target_repetition'])
    selector = PromptSelector(
        target_tracker=tracker,
        min_pressure=CONFIG['dataset']['min_pressure_threshold'],
    )

    final_by_category: Dict[str, List[ValidatedPrompt]] = {}
    combined: List[ValidatedPrompt] = []

    categories = CONFIG['dataset']['categories']
    for category in categories:
        candidates = candidates_by_category.get(category, [])
        if not candidates:
            final_by_category[category] = []
            continue

        validated = validate_and_enrich_prompts(
            candidates,
            validator,
            model_wrapper=model_wrapper,
            output_file=validated_dir / f"{category}_validated.jsonl",
        )

        if category in ("creative", "ood"):
            grouped = group_by_target(validated)
            selected = []
            for group in grouped.values():
                best = selector.select_best_from_candidates(group)
                if best:
                    selected.append(best)
        else:
            selected = [vp for vp in validated if vp.validation.is_accepted(vp.candidate.target_word)]

        for vp in selected:
            if vp.s_score == 0:
                vp.compute_s_score()

        # Tau-lowering loop with bin balancing (per spec Section 3.8)
        # Start with min_pressure_threshold and lower by 0.05 until we have enough prompts
        initial_tau = CONFIG['dataset'].get('min_pressure_threshold', 0.20)
        tau = initial_tau
        tau_step = 0.05
        min_tau = 0.05
        target_per_bin = 100
        num_bins = 5
        bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Track bin shortfalls for metadata
        bin_shortfalls_category: Dict[str, int] = {}
        final_tau = tau
        final_selected: List[ValidatedPrompt] = []
        last_balanced: List[ValidatedPrompt] = []
        last_bin_shortfalls: Dict[str, int] = {}
        last_tau_with_balanced: Optional[float] = None

        psem_available = any(vp.p_sem > 0 for vp in selected)
        if selected and not psem_available:
            logger.warning("p_sem unavailable for category %s; proceeding without pressure gating", category)

        while tau >= min_tau:
            if not selected:
                break

            # Gate by pressure threshold unless p_sem unavailable
            if not psem_available:
                gated = selected
            else:
                gated = [vp for vp in selected if vp.p_sem >= tau]

            if not gated:
                tau -= tau_step
                continue

            # Bin balance: assign prompts to bins, keep top 100 by s_score per bin
            bins: Dict[int, List[ValidatedPrompt]] = {i: [] for i in range(num_bins)}
            for vp in gated:
                # Clamp p_sem to [0, 1] and compute bin index
                p_clamped = max(0.0, min(1.0, vp.p_sem))
                bin_idx = min(int(p_clamped * num_bins), num_bins - 1)
                bins[bin_idx].append(vp)

            # Select top 100 per bin by s_score (no random sampling)
            balanced: List[ValidatedPrompt] = []
            bin_shortfalls_current: Dict[str, int] = {}
            for bin_idx in range(num_bins):
                bin_prompts = bins[bin_idx]
                bin_prompts.sort(key=lambda p: -p.s_score)
                kept = bin_prompts[:target_per_bin]
                balanced.extend(kept)
                shortfall = target_per_bin - len(kept)
                if shortfall > 0:
                    bin_label = f"{bin_edges[bin_idx]:.1f}-{bin_edges[bin_idx+1]:.1f}"
                    bin_shortfalls_current[bin_label] = shortfall

            last_balanced = balanced
            last_bin_shortfalls = bin_shortfalls_current
            last_tau_with_balanced = tau

            if len(balanced) >= prompts_per_category:
                # We have enough; trim to exactly 500 by overall s_score
                balanced.sort(key=lambda p: -p.s_score)
                final_selected = balanced[:prompts_per_category]
                final_tau = tau
                bin_shortfalls_category = bin_shortfalls_current
                break

            # Not enough prompts; lower tau and retry
            tau -= tau_step

        if not final_selected:
            if not last_balanced:
                raise RuntimeError(
                    f"Category {category}: no prompts available after gating. "
                    "Check validation filters or model availability."
                )
            if len(last_balanced) < prompts_per_category:
                raise RuntimeError(
                    f"Category {category}: could not reach {prompts_per_category} prompts even with tau={min_tau}. "
                    f"Only {len(last_balanced)} prompts available after bin balancing. "
                    "Hard halt per spec Section 14.4."
                )
            last_balanced.sort(key=lambda p: -p.s_score)
            final_selected = last_balanced[:prompts_per_category]
            final_tau = last_tau_with_balanced if last_tau_with_balanced is not None else min_tau
            bin_shortfalls_category = last_bin_shortfalls
        
        # Store final tau for this category in metadata (will be used later)
        if not hasattr(build_dataset, '_tau_per_category'):
            build_dataset._tau_per_category = {}
        build_dataset._tau_per_category[category] = final_tau
        
        if not hasattr(build_dataset, '_bin_shortfalls'):
            build_dataset._bin_shortfalls = {}
        if bin_shortfalls_category:
            build_dataset._bin_shortfalls[category] = bin_shortfalls_category

        final_selected = selector.filter_by_target_repetition(final_selected)
        if len(final_selected) < prompts_per_category:
            raise RuntimeError(
                f"Category {category}: target repetition filter reduced prompts to {len(final_selected)} "
                f"(need {prompts_per_category}). Increase candidate pool or adjust max_target_repetition."
            )
        final_selected.sort(key=lambda p: -p.s_score)
        final_by_category[category] = final_selected[:prompts_per_category]
        combined.extend(final_by_category[category])

        logger.info(
            "Category %s: %d selected (from %d candidates, tau=%.2f)",
            category,
            len(final_by_category[category]),
            len(candidates),
            final_tau,
        )

    _write_jsonl([vp.to_dict() for vp in combined], validated_dir / "prompts.jsonl")
    for category, prompts in final_by_category.items():
        _write_jsonl([vp.to_dict() for vp in prompts], validated_dir / f"{category}_prompts.jsonl")

    # Write prompts.csv with full schema
    import hashlib
    import pandas as pd
    from .prompt_builder import build_prompt

    # Compute p1 (negative P_sem) for all prompts
    # Try to load model, but don't crash if unavailable
    p1_values: Dict[str, float] = {}
    try:
        from .metrics_psem import compute_p_sem_for_prompt
        wrapper = model_wrapper or ModelWrapper.get_instance()
        if wrapper.model is None:
            wrapper.load()

        logger.info("Computing p1 (negative P_sem) for %d prompts...", len(combined))
        for vp in combined:
            prompt_id = _compute_prompt_id(
                vp.candidate.category,
                vp.candidate.target_word_normalized,
                vp.candidate.question_text,
                vp.candidate.target_word
            )
            try:
                neg_prompt = build_prompt(
                    vp.candidate.question_text,
                    vp.candidate.target_word,
                    "negative"
                )
                p1_values[prompt_id] = compute_p_sem_for_prompt(
                    neg_prompt,
                    vp.candidate.target_word,
                    wrapper.model,
                    wrapper.tokenizer,
                )
            except Exception as e:
                logger.warning("p1 computation failed for %s: %s", prompt_id, e)
                p1_values[prompt_id] = 0.0

    except Exception as e:
        logger.warning("Model unavailable for p1 computation: %s. Setting p1=0.0 for all.", e)
        for vp in combined:
            prompt_id = _compute_prompt_id(
                vp.candidate.category,
                vp.candidate.target_word_normalized,
                vp.candidate.question_text,
                vp.candidate.target_word
            )
            p1_values[prompt_id] = 0.0

    # Get pressure bins from config
    pressure_bins = CONFIG['dataset'].get('pressure_bins', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def compute_p0_bin(p0: float) -> str:
        """Compute p0_bin string from p0 value."""
        # Clamp to [0, 1]
        p0_clamped = max(0.0, min(1.0, p0))
        for i in range(len(pressure_bins) - 1):
            lower = pressure_bins[i]
            upper = pressure_bins[i + 1]
            # Last bin includes 1.0
            if i == len(pressure_bins) - 2:
                if lower <= p0_clamped <= upper:
                    return f"{lower:.1f}-{upper:.1f}"
            else:
                if lower <= p0_clamped < upper:
                    return f"{lower:.1f}-{upper:.1f}"
        # Fallback: last bin
        return f"{pressure_bins[-2]:.1f}-{pressure_bins[-1]:.1f}"

    csv_rows = []
    for vp in combined:
        # question_text is the raw cloze question without instruction
        question_text = vp.candidate.question_text
        if not question_text or not question_text.strip():
            raise ValueError(
                f"Missing question_text for prompt {vp.candidate.target_word}. "
                "All prompts must have a valid question_text."
            )

        prompt_id = _compute_prompt_id(
            vp.candidate.category,
            vp.candidate.target_word_normalized,
            question_text,
            vp.candidate.target_word
        )

        # prompt_text is the full baseline prompt
        prompt_text = build_prompt(question_text, vp.candidate.target_word, "baseline")

        # p0 is baseline P_sem (already computed as p_sem)
        p0 = vp.p_sem
        p1 = p1_values.get(prompt_id, 0.0)
        p0_bin = compute_p0_bin(p0)

        # validation_json_ref: reference to validated JSONL file
        validation_json_ref = f"{vp.candidate.category}_validated.jsonl#prompt_id={prompt_id}"

        csv_rows.append({
            "prompt_id": prompt_id,
            "question_text": question_text,
            "prompt_text": prompt_text,
            "category": vp.candidate.category,
            "target_word": vp.candidate.target_word,
            "target_word_normalized": vp.candidate.target_word_normalized,
            "prompt_style_id": vp.candidate.prompt_style_id,
            "source_trace": vp.candidate.source_trace,
            "validation_json_ref": validation_json_ref,
            "p0": p0,
            "p1": p1,
            "p0_bin": p0_bin,
            "v_score": vp.validation.v_score if vp.validation else 0,
            "p_sem": p0,  # Alias of p0 for backward compatibility
            "s_score": vp.s_score,
        })

    # Validate all rows have question_text
    for row in csv_rows:
        if not row.get("question_text"):
            raise ValueError(
                f"question_text is empty for prompt_id={row.get('prompt_id')}. "
                "Cannot write prompts.csv with missing question_text."
            )

    # Write prompts.csv
    prompts_df = pd.DataFrame(csv_rows)
    prompts_csv_path = data_root / "prompts.csv"
    prompts_csv_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_df.to_csv(prompts_csv_path, index=False)
    logger.info(f"Wrote {len(prompts_df)} prompts to {prompts_csv_path}")

    # Write prompts_metadata.json
    from datetime import datetime

    csv_content = prompts_csv_path.read_bytes()
    dataset_hash = hashlib.sha256(csv_content).hexdigest()

    counts_by_category: Dict[str, int] = {}
    counts_by_p0_bin: Dict[str, int] = {}
    for row in csv_rows:
        cat = row["category"]
        counts_by_category[cat] = counts_by_category.get(cat, 0) + 1
        bin_str = row["p0_bin"]
        counts_by_p0_bin[bin_str] = counts_by_p0_bin.get(bin_str, 0) + 1

    # Compute tau_per_category
    # If tau varies per category, use that; otherwise set all to min_pressure_threshold
    min_tau = CONFIG['dataset'].get('min_pressure_threshold', 0.2)
    tau_per_category: Dict[str, float] = {}
    try:
        # Try to get category-specific tau if available
        tau_config = CONFIG['dataset'].get('tau_per_category', {})
        if tau_config:
            tau_per_category = dict(tau_config)
        else:
            # Use default for all categories
            for cat in counts_by_category:
                tau_per_category[cat] = min_tau
    except Exception as e:
        logger.warning("Could not get tau_per_category: %s. Using default.", e)
        for cat in counts_by_category:
            tau_per_category[cat] = min_tau

    # Get sampling seeds
    sampling_seeds: Dict[str, int] = {}
    try:
        sampling_seeds = dict(CONFIG.get('seeds', {'python': 42, 'numpy': 42, 'torch': 42}))
    except Exception:
        sampling_seeds = {'python': 42, 'numpy': 42, 'torch': 42}

    # DeepSeek model IDs
    deepseek_models = {
        "generator": CONFIG.get('deepseek', {}).get('generator_model', 'deepseek-chat'),
        "validator": CONFIG.get('deepseek', {}).get('validator_model', 'deepseek-reasoner'),
    }

    # Use actual tau values computed during category processing
    if hasattr(build_dataset, '_tau_per_category'):
        tau_per_category = build_dataset._tau_per_category
    
    # Get bin shortfalls if any were recorded
    bin_shortfalls: Dict[str, Dict[str, int]] = {}
    if hasattr(build_dataset, '_bin_shortfalls'):
        bin_shortfalls = build_dataset._bin_shortfalls
    
    # Check for gating failures (categories with <500 prompts)
    gating_failures: List[str] = []
    for cat, count in counts_by_category.items():
        if count < prompts_per_category:
            gating_failures.append(cat)

    metadata = {
        "prompt_id_type": "string_key",
        "prompt_id_format": "{category}_{target_word_normalized}_{hash8}",
        "total_count": len(csv_rows),
        "counts_by_category": counts_by_category,
        "counts_by_p0_bin": counts_by_p0_bin,
        "tau_per_category": tau_per_category,
        "bin_shortfalls": bin_shortfalls,
        "gating_failures": gating_failures,
        "sampling_seeds": sampling_seeds,
        "deepseek_models": deepseek_models,
        "timestamps": {
            "generated_at": datetime.now().isoformat(),
        },
        "pressure_bins": pressure_bins,
        "min_pressure_threshold": min_tau,
        "dataset_hash": dataset_hash,
    }

    metadata_path = data_root / "prompts_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote prompts_metadata.json to {metadata_path}")

    return final_by_category


# ============================================================================
# SELF-TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATASET PIPELINE TESTS")
    print("=" * 60)
    
    # Test 1: prompt_id hash stability
    print("\n1. Testing prompt_id hash stability:")
    
    # Same inputs should produce same hash
    test_cases = [
        ("facts", "paris", "The capital of France is ____.", "Paris"),
        ("idioms", "bucket", "Kick the ____.", "bucket"),
        ("creative", "space", "The astronaut gazed at the endless ____.", "space"),
    ]
    
    all_passed = True
    for category, target_norm, question, target in test_cases:
        id1 = _compute_prompt_id(category, target_norm, question, target)
        id2 = _compute_prompt_id(category, target_norm, question, target)
        if id1 == id2:
            print(f"   ✅ prompt_id stable for {category}_{target_norm}: {id1}")
        else:
            print(f"   ❌ prompt_id NOT stable: {id1} != {id2}")
            all_passed = False
    
    # Different inputs should produce different hashes
    id1 = _compute_prompt_id("facts", "paris", "The capital of France is ____.", "Paris")
    id2 = _compute_prompt_id("facts", "paris", "The capital of Germany is ____.", "Paris")
    if id1 != id2:
        print(f"   ✅ Different questions produce different hashes")
    else:
        print(f"   ❌ Different questions produce SAME hash (collision)")
        all_passed = False
    
    # Verify format
    id1 = _compute_prompt_id("facts", "paris", "Test question", "Paris")
    parts = id1.split("_")
    if len(parts) == 3 and parts[0] == "facts" and parts[1] == "paris" and len(parts[2]) == 8:
        print(f"   ✅ prompt_id format correct: {id1}")
    else:
        print(f"   ❌ prompt_id format incorrect: {id1}")
        all_passed = False
    
    print("\n" + "=" * 60)
    print(f"Dataset pipeline tests {'PASSED' if all_passed else 'FAILED'}!")
    print("=" * 60)
