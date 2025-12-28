"""
dataset_pipeline.py - End-to-end dataset construction for Module 2.
"""

import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .api_clients import OpenAIClient
from .config import CONFIG, get_base_paths
from .data_mining import CandidatePrompt, DatasetGenerator
from .utils import ModelWrapper
from .validator import (
    PromptSelector,
    PromptValidator,
    TargetTracker,
    ValidatedPrompt,
    compute_semantic_pressure,
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


def _normalize_question_key(question: str) -> str:
    return " ".join(question.lower().split())


def _dedupe_candidates(candidates: List[CandidatePrompt]) -> List[CandidatePrompt]:
    deduped: List[CandidatePrompt] = []
    seen = set()
    for cand in candidates:
        key = (_normalize_question_key(cand.question_text), cand.target_word_normalized)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def _dedupe_validated(validated: List[ValidatedPrompt]) -> List[ValidatedPrompt]:
    deduped: List[ValidatedPrompt] = []
    seen = set()
    for vp in validated:
        key = (_normalize_question_key(vp.candidate.question_text), vp.candidate.target_word_normalized)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(vp)
    return deduped


def _count_accepted(validated: List[ValidatedPrompt]) -> int:
    return sum(
        1 for vp in validated
        if vp.validation.is_accepted(vp.candidate.target_word)
    )


def _round_up_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return int(math.ceil(value / multiple) * multiple)


def _common_sense_relation_counts(candidates: List[CandidatePrompt]) -> Dict[str, int]:
    counts = {
        "UsedFor": 0,
        "MadeOf": 0,
        "HasProperty": 0,
        "HasPart": 0,
        "AtLocation": 0,
        "CapableOf": 0,
        "UsedBy": 0,
        "Requires": 0,
        "Contains": 0,
        "WornOn": 0,
    }
    for cand in candidates:
        relation = cand.raw_data.get("relation")
        if not relation and cand.prompt_style_id:
            if cand.prompt_style_id.startswith("C1_"):
                relation = "UsedFor"
            elif cand.prompt_style_id.startswith("C2_"):
                relation = "MadeOf"
            elif cand.prompt_style_id.startswith("C3_"):
                relation = "HasProperty"
            elif cand.prompt_style_id.startswith("C4_"):
                relation = "HasPart"
            elif cand.prompt_style_id.startswith("C5_"):
                relation = "AtLocation"
            elif cand.prompt_style_id.startswith("C6_"):
                relation = "CapableOf"
            elif cand.prompt_style_id.startswith("C7_"):
                relation = "UsedBy"
            elif cand.prompt_style_id.startswith("C8_"):
                relation = "Requires"
            elif cand.prompt_style_id.startswith("C9_"):
                relation = "Contains"
            elif cand.prompt_style_id.startswith("C10_"):
                relation = "WornOn"
        if relation in counts:
            counts[relation] += 1
    return counts


def _common_sense_relation_targets(gap: int, counts: Dict[str, int]) -> Dict[str, int]:
    relation_types = [
        "UsedFor",
        "MadeOf",
        "HasProperty",
        "HasPart",
        "AtLocation",
        "CapableOf",
        "UsedBy",
        "Requires",
        "Contains",
        "WornOn",
    ]
    targets = {rel: 0 for rel in relation_types}
    for _ in range(gap):
        rel = min(relation_types, key=lambda r: counts.get(r, 0) + targets[r])
        targets[rel] += 1
    return targets


def _log_common_sense_diversity(candidates: List[CandidatePrompt]) -> None:
    if not candidates:
        return
    unique_targets = len({c.target_word_normalized for c in candidates})
    subjects = set()
    for cand in candidates:
        subject = cand.raw_data.get("subject")
        if subject:
            subjects.add(" ".join(str(subject).lower().split()))
    relation_counts = _common_sense_relation_counts(candidates)
    logger.info(
        "Common sense diversity: %d candidates, %d unique targets, %d unique subjects, relations=%s",
        len(candidates),
        unique_targets,
        len(subjects),
        relation_counts,
    )


def _log_category_diversity(category: str, candidates: List[CandidatePrompt]) -> None:
    if not candidates:
        return
    unique_targets = len({c.target_word_normalized for c in candidates})
    unique_questions = len({_normalize_question_key(c.question_text) for c in candidates})
    logger.info(
        "Category %s diversity: %d candidates, %d unique targets, %d unique questions",
        category,
        len(candidates),
        unique_targets,
        unique_questions,
    )


def group_by_target(
    validated_prompts: List[ValidatedPrompt],
) -> Dict[str, List[ValidatedPrompt]]:
    groups: Dict[str, List[ValidatedPrompt]] = {}
    for vp in validated_prompts:
        key = vp.candidate.target_word_normalized
        groups.setdefault(key, []).append(vp)
    return groups


def _load_validated_prompts(path: Path) -> List[ValidatedPrompt]:
    """Load ValidatedPrompt JSONL entries from disk."""
    prompts: List[ValidatedPrompt] = []
    skipped = 0
    if not path.exists():
        raise FileNotFoundError(f"Validated prompt file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            prompts.append(ValidatedPrompt.from_dict(obj))
    if skipped:
        logger.warning("Skipped %d invalid JSONL lines in %s", skipped, path)
    return prompts


def _load_candidate_prompts(path: Path) -> List[CandidatePrompt]:
    """Load CandidatePrompt JSONL entries from disk."""
    prompts: List[CandidatePrompt] = []
    skipped = 0
    if not path.exists():
        raise FileNotFoundError(f"Candidate prompt file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            prompts.append(CandidatePrompt.from_dict(obj))
    if skipped:
        logger.warning("Skipped %d invalid JSONL lines in %s", skipped, path)
    return prompts


def _select_prompts_for_category(
    category: str,
    validated: List[ValidatedPrompt],
    selector: PromptSelector,
    prompts_per_category: int,
) -> Tuple[List[ValidatedPrompt], float, Dict[str, int]]:
    """
    Select, gate, and bin-balance prompts for a single category.
    """
    if not validated:
        raise RuntimeError(f"No validated prompts available for category {category}")

    if category in ("creative", "ood"):
        grouped = group_by_target(validated)
        selected = []
        for group in grouped.values():
            best = selector.select_best_from_candidates(group)
            if best:
                selected.append(best)
    else:
        selected = [
            vp for vp in validated
            if vp.validation.is_accepted(vp.candidate.target_word)
        ]

    all_ranked = list(validated)
    for vp in all_ranked:
        if vp.s_score == 0:
            vp.compute_s_score()

    def _prompt_id(vp: ValidatedPrompt) -> str:
        if vp.prompt_id:
            return vp.prompt_id
        return _compute_prompt_id(
            vp.candidate.category,
            vp.candidate.target_word_normalized,
            vp.candidate.question_text,
            vp.candidate.target_word,
        )

    ranked = sorted(selected, key=lambda p: -p.s_score)
    all_ranked.sort(key=lambda p: -p.s_score)

    # Tau-lowering loop with bin balancing (per spec Section 3.9)
    initial_tau = CONFIG['dataset'].get('min_pressure_threshold', 0.20)
    tau = initial_tau
    tau_step = 0.05
    min_tau = 0.05
    target_per_bin = 100
    num_bins = 5
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    final_selected: List[ValidatedPrompt] = []
    last_balanced: List[ValidatedPrompt] = []
    last_bin_shortfalls: Dict[str, int] = {}
    last_tau_with_balanced: Optional[float] = None
    bin_shortfalls_category: Dict[str, int] = {}
    final_tau = tau

    while tau >= min_tau:
        if not selected:
            break

        gated = [vp for vp in selected if vp.p_sem >= tau]
        if not gated:
            tau -= tau_step
            continue

        bins: Dict[int, List[ValidatedPrompt]] = {i: [] for i in range(num_bins)}
        for vp in gated:
            p_clamped = max(0.0, min(1.0, vp.p_sem))
            bin_idx = min(int(p_clamped * num_bins), num_bins - 1)
            bins[bin_idx].append(vp)

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
            balanced.sort(key=lambda p: -p.s_score)
            final_selected = balanced[:prompts_per_category]
            final_tau = tau
            bin_shortfalls_category = bin_shortfalls_current
            break

        tau -= tau_step

    if not final_selected:
        if last_balanced:
            last_balanced.sort(key=lambda p: -p.s_score)
            final_selected = last_balanced
            final_tau = last_tau_with_balanced if last_tau_with_balanced is not None else min_tau
            bin_shortfalls_category = last_bin_shortfalls
        else:
            final_selected = ranked
            final_tau = min_tau
            bin_shortfalls_category = last_bin_shortfalls

    if len(final_selected) < prompts_per_category:
        chosen_ids = {_prompt_id(vp) for vp in final_selected}
        for vp in ranked:
            pid = _prompt_id(vp)
            if pid in chosen_ids:
                continue
            final_selected.append(vp)
            chosen_ids.add(pid)
            if len(final_selected) >= prompts_per_category:
                break

    prioritized = final_selected[:]
    prioritized_ids = {_prompt_id(vp) for vp in prioritized}
    for vp in ranked:
        pid = _prompt_id(vp)
        if pid in prioritized_ids:
            continue
        prioritized.append(vp)
        prioritized_ids.add(pid)

    filtered: List[ValidatedPrompt] = []
    for vp in prioritized:
        if selector.target_tracker.register(vp.candidate.target_word_normalized, vp.candidate.category):
            filtered.append(vp)
            if len(filtered) >= prompts_per_category:
                break

    if len(filtered) < prompts_per_category:
        logger.warning(
            "Category %s: only %d prompts after repetition filtering (target %d). "
            "Proceeding without regeneration.",
            category,
            len(filtered),
            prompts_per_category,
        )
        chosen_ids = {_prompt_id(vp) for vp in filtered}
        for vp in all_ranked:
            pid = _prompt_id(vp)
            if pid in chosen_ids:
                continue
            filtered.append(vp)
            chosen_ids.add(pid)
            if len(filtered) >= prompts_per_category:
                break
        if len(filtered) < prompts_per_category:
            logger.warning(
                "Category %s: only %d prompts available after fallback fill (target %d).",
                category,
                len(filtered),
                prompts_per_category,
            )

    filtered.sort(key=lambda p: -p.s_score)

    return filtered[:prompts_per_category], final_tau, bin_shortfalls_category


def build_dataset(
    prompts_per_category: int = CONFIG['dataset']['prompts_per_category'],
    candidate_multiplier: int = CONFIG['dataset'].get('candidate_multiplier', 2),
    candidates_per_target: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    openai_client: Optional[OpenAIClient] = None,
    model_wrapper: Optional[ModelWrapper] = None,
    skip_generated: bool = False,
    use_fallback: bool = True,
    output_root: Optional[Path] = None,
) -> Dict[str, List[ValidatedPrompt]]:
    """
    Generate, validate, score, and select prompts for Module 2.
    """
    if use_fallback:
        logger.warning(
            "GPT-5.2 fallback enabled: source gaps will be filled via batch generation."
        )
    paths = get_base_paths()
    data_root = output_root or paths['data_root']
    cache_dir = cache_dir or (data_root / "raw")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    if openai_client is None:
        openai_client = OpenAIClient()
    if openai_client.request_log_path is None:
        openai_client.request_log_path = str(data_root / "gpt5_requests.jsonl")

    candidates_dir = data_root / "candidates"
    validated_dir = data_root / "validated"

    generator = DatasetGenerator(cache_dir=cache_dir, openai_client=openai_client)
    candidates_by_category = generator.generate_all(
        prompts_per_category=prompts_per_category,
        candidate_multiplier=candidate_multiplier,
        candidates_per_target=candidates_per_target,
        skip_generated=skip_generated,
        use_fallback=use_fallback,
    )

    save_candidates(candidates_by_category, candidates_dir)

    validator = PromptValidator(openai_client)
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
        final_selected, final_tau, bin_shortfalls_category = _select_prompts_for_category(
            category=category,
            validated=validated,
            selector=selector,
            prompts_per_category=prompts_per_category,
        )

        if not hasattr(build_dataset, '_tau_per_category'):
            build_dataset._tau_per_category = {}
        build_dataset._tau_per_category[category] = final_tau

        if not hasattr(build_dataset, '_bin_shortfalls'):
            build_dataset._bin_shortfalls = {}
        if bin_shortfalls_category:
            build_dataset._bin_shortfalls[category] = bin_shortfalls_category

        final_by_category[category] = final_selected
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

    tau_per_category = getattr(build_dataset, "_tau_per_category", {})
    bin_shortfalls = getattr(build_dataset, "_bin_shortfalls", {})
    _write_prompts_outputs(
        combined=combined,
        data_root=data_root,
        prompts_per_category=prompts_per_category,
        model_wrapper=model_wrapper,
        tau_per_category=tau_per_category,
        bin_shortfalls=bin_shortfalls,
    )

    return final_by_category


def build_r1_validated_candidates(
    prompts_per_category: int = CONFIG['dataset']['prompts_per_category'],
    candidate_multiplier: int = CONFIG['dataset'].get('candidate_multiplier', 2),
    candidates_per_target: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    openai_client: Optional[OpenAIClient] = None,
    skip_generated: bool = False,
    output_root: Optional[Path] = None,
    categories: Optional[List[str]] = None,
    resume: bool = True,
    min_accepted_per_category: Optional[int] = None,
    acceptance_buffer: float = 1.5,
    max_rounds: int = 5,
) -> Dict[str, List[ValidatedPrompt]]:
    """
    Generate candidate prompts and validate them with GPT-5.2 only.

    This stage does NOT compute P_sem (Qwen). It builds candidate pools from
    primary sources, fills gaps via GPT-5.2 batch generation, then validates
    all remaining candidates in a single GPT-5.2 batch.
    """
    paths = get_base_paths()
    data_root = output_root or paths['data_root']
    cache_dir = cache_dir or (data_root / "raw")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    if openai_client is None:
        openai_client = OpenAIClient()
    if openai_client.request_log_path is None:
        openai_client.request_log_path = str(data_root / "gpt5_requests.jsonl")

    candidates_dir = data_root / "candidates"
    validated_dir = data_root / "validated"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    validated_dir.mkdir(parents=True, exist_ok=True)

    categories = categories or CONFIG['dataset']['categories']

    k = candidates_per_target or CONFIG['dataset']['candidates_per_target']
    base_count = max(1, int(prompts_per_category * candidate_multiplier))
    creative_ood_targets = CONFIG['dataset'].get('creative_ood_target_count', prompts_per_category)
    creative_ood_count = max(1, int(creative_ood_targets * k))
    desired_counts = {cat: base_count for cat in categories}
    for category in ("creative", "ood"):
        if category in desired_counts:
            desired_counts[category] = creative_ood_count

    generator = DatasetGenerator(cache_dir=cache_dir, openai_client=openai_client)
    validator = PromptValidator(openai_client)

    candidates_by_category: Dict[str, List[CandidatePrompt]] = {}
    validated_by_category: Dict[str, List[ValidatedPrompt]] = {}
    validated_ids_by_category: Dict[str, set] = {}

    # Load existing candidates and validations if resuming.
    for category in categories:
        candidates_path = candidates_dir / f"{category}.jsonl"
        validated_path = validated_dir / f"{category}_validated.jsonl"

        candidates: List[CandidatePrompt] = []
        validated: List[ValidatedPrompt] = []

        if resume and candidates_path.exists():
            candidates = _load_candidate_prompts(candidates_path)
            candidates = _dedupe_candidates(candidates)

        if resume and validated_path.exists():
            validated = _load_validated_prompts(validated_path)
            validated = _dedupe_validated(validated)
            if not candidates:
                candidates = [vp.candidate for vp in validated]

        candidates_by_category[category] = candidates
        validated_by_category[category] = validated
        validated_ids_by_category[category] = {
            vp.prompt_id or vp.validation.prompt_id for vp in validated
        }

    # Generate from primary sources first.
    for category in categories:
        desired = desired_counts.get(category, base_count)
        candidates = candidates_by_category[category]
        if len(candidates) >= desired:
            continue

        source_candidates: List[CandidatePrompt] = []
        try:
            if category == "idioms":
                source_candidates = generator.idiom_gen.generate_candidates(desired - len(candidates))
            elif category == "facts":
                source_candidates = generator.fact_gen.generate_candidates(desired - len(candidates))
            elif category == "common_sense":
                source_candidates = generator.common_sense_gen.generate_candidates(desired - len(candidates))
            elif category in ("creative", "ood"):
                source_candidates = []
        except Exception as e:
            logger.warning("Primary source failed for %s: %s", category, e)
            source_candidates = []

        if source_candidates:
            candidates.extend(source_candidates)
            candidates = _dedupe_candidates(candidates)
            candidates_by_category[category] = candidates

    # Fill gaps via GPT-5.2 batch generation.
    for category in categories:
        desired = desired_counts.get(category, base_count)
        candidates = candidates_by_category[category]
        gap = desired - len(candidates)
        if gap <= 0:
            continue

        if skip_generated and category in ("creative", "ood"):
            logger.warning("Skipping GPT-5.2 generation for %s; gap remains %d.", category, gap)
            continue

        if category == "common_sense":
            relation_counts = _common_sense_relation_counts(candidates)
            relation_targets = _common_sense_relation_targets(gap, relation_counts)
            logger.warning(
                "Common sense: only %d ConceptNet candidates; filling %d via GPT-5.2 batch fallback.",
                len(candidates),
                gap,
            )
            new_candidates = generator.fallback_gen.generate_for_category(
                "common_sense",
                n=gap,
                relation_targets=relation_targets,
            )
        elif category == "creative":
            batch_count = _round_up_multiple(gap, k)
            new_candidates = generator.creative_gen.generate_candidates(
                batch_count,
                candidates_per_target=k,
            )
        elif category == "ood":
            batch_count = _round_up_multiple(gap, k)
            new_candidates = generator.ood_gen.generate_candidates(
                batch_count,
                candidates_per_target=k,
            )
        else:
            logger.warning(
                "Category %s: only %d candidates; filling %d via GPT-5.2 batch fallback.",
                category,
                len(candidates),
                gap,
            )
            new_candidates = generator.fallback_gen.generate_for_category(category, n=gap)

        candidates.extend(new_candidates)
        candidates = _dedupe_candidates(candidates)
        candidates_by_category[category] = candidates

        if len(candidates) < desired:
            logger.warning(
                "Category %s: only %d candidates after fallback (target %d). Proceeding without regeneration.",
                category,
                len(candidates),
                desired,
            )

    # Write candidates to disk.
    for category, candidates in candidates_by_category.items():
        candidates_path = candidates_dir / f"{category}.jsonl"
        _write_jsonl([c.to_dict() for c in candidates], candidates_path)
        _log_category_diversity(category, candidates)
        if category == "common_sense":
            _log_common_sense_diversity(candidates)

    # Validate all remaining candidates in one GPT-5.2 batch.
    to_validate: List[CandidatePrompt] = []
    for category, candidates in candidates_by_category.items():
        existing_ids = validated_ids_by_category.get(category, set())
        for cand in candidates:
            prompt_id = validator._build_prompt_id(cand)
            if prompt_id not in existing_ids:
                to_validate.append(cand)

    if to_validate:
        validated_new = validate_and_enrich_prompts(
            to_validate,
            validator,
            model_wrapper=None,
            output_file=None,
            compute_pressure=False,
            append=False,
            write_output=False,
        )
        for vp in validated_new:
            validated_by_category.setdefault(vp.candidate.category, []).append(vp)

    # Deduplicate and write validated outputs.
    min_accepted_per_category = min_accepted_per_category or prompts_per_category
    combined: List[ValidatedPrompt] = []
    for category in categories:
        validated = _dedupe_validated(validated_by_category.get(category, []))
        validated_by_category[category] = validated

        accepted_count = _count_accepted(validated)
        if accepted_count < min_accepted_per_category:
            logger.warning(
                "Category %s: ended with %d validated and %d accepted (target %d). "
                "Proceeding without regeneration.",
                category,
                len(validated),
                accepted_count,
                min_accepted_per_category,
            )

        validated_path = validated_dir / f"{category}_validated.jsonl"
        _write_jsonl([vp.to_dict() for vp in validated], validated_path)
        combined.extend(validated)

    if combined:
        _write_jsonl([vp.to_dict() for vp in combined], validated_dir / "all_validated.jsonl")

    all_candidates: List[CandidatePrompt] = []
    for category in categories:
        path = candidates_dir / f"{category}.jsonl"
        if path.exists():
            all_candidates.extend(_load_candidate_prompts(path))
    if all_candidates:
        _write_jsonl([c.to_dict() for c in all_candidates], candidates_dir / "all_candidates.jsonl")

    logger.info(
        "Wrote GPT-5.2 validated candidates for %d categories to %s",
        len(validated_by_category),
        validated_dir,
    )
    return validated_by_category


def finalize_dataset_with_psem(
    validated_dir: Optional[Path] = None,
    output_root: Optional[Path] = None,
    model_wrapper: Optional[ModelWrapper] = None,
    prompts_per_category: int = CONFIG['dataset']['prompts_per_category'],
) -> Dict[str, List[ValidatedPrompt]]:
    """
    Compute P_sem (P0/P1), select final prompts, and write prompts.csv/metadata.

    Expects reasoner-validated JSONL files already present under data/validated.
    """
    paths = get_base_paths()
    data_root = output_root or paths['data_root']
    validated_dir = validated_dir or (data_root / "validated")

    categories = CONFIG['dataset']['categories']
    validated_by_category: Dict[str, List[ValidatedPrompt]] = {}
    all_validated: List[ValidatedPrompt] = []

    for category in categories:
        path = validated_dir / f"{category}_validated.jsonl"
        validated = _load_validated_prompts(path)
        if not validated:
            raise RuntimeError(
                f"No validated prompts found for category {category} at {path}"
            )
        validated_by_category[category] = validated
        all_validated.extend(validated)

    compute_semantic_pressure(all_validated, model_wrapper=model_wrapper)

    selector = PromptSelector(
        target_tracker=TargetTracker(
            max_repetition=CONFIG['dataset']['max_target_repetition']
        ),
        min_pressure=CONFIG['dataset']['min_pressure_threshold'],
    )

    final_by_category: Dict[str, List[ValidatedPrompt]] = {}
    combined: List[ValidatedPrompt] = []
    tau_per_category: Dict[str, float] = {}
    bin_shortfalls: Dict[str, Dict[str, int]] = {}

    for category in categories:
        final_selected, final_tau, bin_shortfalls_category = _select_prompts_for_category(
            category=category,
            validated=validated_by_category[category],
            selector=selector,
            prompts_per_category=prompts_per_category,
        )
        final_by_category[category] = final_selected
        combined.extend(final_selected)
        tau_per_category[category] = final_tau
        if bin_shortfalls_category:
            bin_shortfalls[category] = bin_shortfalls_category

    _write_jsonl([vp.to_dict() for vp in combined], validated_dir / "prompts.jsonl")
    for category, prompts in final_by_category.items():
        _write_jsonl([vp.to_dict() for vp in prompts], validated_dir / f"{category}_prompts.jsonl")

    _write_prompts_outputs(
        combined=combined,
        data_root=data_root,
        prompts_per_category=prompts_per_category,
        model_wrapper=model_wrapper,
        tau_per_category=tau_per_category,
        bin_shortfalls=bin_shortfalls,
    )

    return final_by_category


def _write_prompts_outputs(
    combined: List[ValidatedPrompt],
    data_root: Path,
    prompts_per_category: int,
    model_wrapper: Optional[ModelWrapper] = None,
    tau_per_category: Optional[Dict[str, float]] = None,
    bin_shortfalls: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    """
    Compute P1, write prompts.csv, and write prompts_metadata.json.
    """
    import hashlib
    import pandas as pd
    from .prompt_builder import build_prompt
    from .metrics_psem import compute_p_sem_for_prompt

    if not combined:
        raise RuntimeError("No prompts provided for prompts.csv output.")

    # Compute p1 (negative P_sem) for all prompts (hard fail on errors)
    p1_values: Dict[str, float] = {}
    wrapper = model_wrapper or ModelWrapper.get_instance()
    if not wrapper.is_loaded:
        wrapper.load()

    failures: List[Dict[str, str]] = []
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
            failures.append({"prompt_id": prompt_id, "error": str(e)})

    if failures:
        sample = failures[:3]
        raise RuntimeError(
            f"p1 computation failed for {len(failures)} prompts. Sample errors: {sample}"
        )

    pressure_bins = CONFIG['dataset'].get('pressure_bins', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def compute_p0_bin(p0: float) -> str:
        """Compute p0_bin string from p0 value."""
        p0_clamped = max(0.0, min(1.0, p0))
        for i in range(len(pressure_bins) - 1):
            lower = pressure_bins[i]
            upper = pressure_bins[i + 1]
            if i == len(pressure_bins) - 2:
                if lower <= p0_clamped <= upper:
                    return f"{lower:.1f}-{upper:.1f}"
            else:
                if lower <= p0_clamped < upper:
                    return f"{lower:.1f}-{upper:.1f}"
        return f"{pressure_bins[-2]:.1f}-{pressure_bins[-1]:.1f}"

    csv_rows = []
    for vp in combined:
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

        prompt_text = build_prompt(question_text, vp.candidate.target_word, "baseline")
        negative_prompt_text = build_prompt(question_text, vp.candidate.target_word, "negative")

        p0 = vp.p_sem
        p1 = p1_values.get(prompt_id, 0.0)
        p0_bin = compute_p0_bin(p0)

        validation_json_ref = f"{vp.candidate.category}_validated.jsonl#prompt_id={prompt_id}"

        csv_rows.append({
            "prompt_id": prompt_id,
            "question_text": question_text,
            "prompt_text": prompt_text,
            "negative_prompt_text": negative_prompt_text,
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
            "p_sem": p0,
            "s_score": vp.s_score,
        })

    for row in csv_rows:
        if not row.get("question_text"):
            raise ValueError(
                f"question_text is empty for prompt_id={row.get('prompt_id')}. "
                "Cannot write prompts.csv with missing question_text."
            )

    prompts_df = pd.DataFrame(csv_rows)
    prompts_csv_path = data_root / "prompts.csv"
    prompts_csv_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_df.to_csv(prompts_csv_path, index=False)
    logger.info(f"Wrote {len(prompts_df)} prompts to {prompts_csv_path}")

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

    min_tau = CONFIG['dataset'].get('min_pressure_threshold', 0.2)
    if not tau_per_category:
        tau_per_category = {}
        for cat in counts_by_category:
            tau_per_category[cat] = min_tau

    sampling_seeds: Dict[str, int] = {}
    try:
        sampling_seeds = dict(CONFIG.get('seeds', {'python': 42, 'numpy': 42, 'torch': 42}))
    except Exception:
        sampling_seeds = {'python': 42, 'numpy': 42, 'torch': 42}

    gpt5_model = CONFIG.get('openai', {}).get('model', 'gpt-5.2-2025-12-11')

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
        "bin_shortfalls": bin_shortfalls or {},
        "gating_failures": gating_failures,
        "sampling_seeds": sampling_seeds,
        "gpt5_model": gpt5_model,
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
