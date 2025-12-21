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
    use_fallback: bool = True,
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

        if any(vp.p_sem > 0 for vp in selected):
            selected = selector.gate_by_pressure(selected)
        else:
            logger.warning("Pressure scores unavailable for %s; skipping pressure gating", category)

        selected = selector.filter_by_target_repetition(selected)
        selected.sort(key=lambda p: -p.s_score)
        final_by_category[category] = selected[:prompts_per_category]
        combined.extend(final_by_category[category])

        logger.info(
            "Category %s: %d selected (from %d candidates)",
            category,
            len(final_by_category[category]),
            len(candidates),
        )

    _write_jsonl([vp.to_dict() for vp in combined], validated_dir / "prompts.jsonl")
    for category, prompts in final_by_category.items():
        _write_jsonl([vp.to_dict() for vp in prompts], validated_dir / f"{category}_prompts.jsonl")

    return final_by_category

