"""
validator.py - Prompt Validation and Selection

This module provides:
- GPT-5.2 validation of prompts (batch workflow)
- Scoring based on validation rubric
- Pressure probing (requires model - Colab only)
- Selection rule implementation
- Global target word tracking

The "Filter Funnel" as described in the implementation plan.
"""

import json
import logging
import math
import re
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Import CandidatePrompt early - needed for type hints before other imports
try:
    from .data_mining import CandidatePrompt
except ImportError:
    from data_mining import CandidatePrompt

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _count_sentences(text: str) -> int:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text or "") if p.strip()]
    return len(parts)


def _strip_non_letters(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text or "").lower()


def _extract_prompt_context(prompt: CandidatePrompt) -> str:
    if prompt.category == "creative":
        raw = prompt.raw_data.get("microstory")
        if raw:
            return str(raw)
    if prompt.category == "ood":
        raw = prompt.raw_data.get("context")
        if raw:
            return str(raw)

    question_text = prompt.question_text or ""
    question_text = question_text.strip()
    if question_text.lower().startswith("fill the blank with one word:"):
        question_text = question_text.split(":", 1)[-1].strip()
    question_text = re.sub(r"\\s+____\\.?$", "", question_text).strip()
    return question_text


def apply_deterministic_filters(prompts: List[CandidatePrompt]) -> Tuple[List[CandidatePrompt], Dict[str, int]]:
    try:
        from .utils import find_word_occurrences
    except ImportError:
        from utils import find_word_occurrences

    stats = {
        "whole_word_leak": 0,
        "stripped_leak": 0,
        "nonalpha_target": 0,
        "creative_sentence_count": 0,
        "ood_sentence_count": 0,
    }
    kept: List[CandidatePrompt] = []

    for prompt in prompts:
        target = prompt.target_word or ""
        if not target.isalpha():
            stats["nonalpha_target"] += 1
            continue

        if find_word_occurrences(target, prompt.question_text or ""):
            stats["whole_word_leak"] += 1
            continue

        stripped = _strip_non_letters(prompt.question_text or "")
        if target.lower() in stripped:
            stats["stripped_leak"] += 1
            continue

        if prompt.category == "creative":
            context = _extract_prompt_context(prompt)
            if _count_sentences(context) != 2:
                stats["creative_sentence_count"] += 1

        if prompt.category == "ood":
            context = _extract_prompt_context(prompt)
            count = _count_sentences(context)
            if count < 1 or count > 2:
                stats["ood_sentence_count"] += 1

        kept.append(prompt)

    return kept, stats

# Import local modules
try:
    from .api_clients import OpenAIClient
    from .config import CONFIG, PROMPT_TEMPLATES, get_base_paths
    from .utils import ModelWrapper
    from .prompt_builder import build_prompt
    from .metrics_psem import compute_p_sem_for_prompt
except ImportError:
    from api_clients import OpenAIClient
    from config import CONFIG, PROMPT_TEMPLATES, get_base_paths
    from utils import ModelWrapper
    from prompt_builder import build_prompt
    from metrics_psem import compute_p_sem_for_prompt


# ============================================================================
# VALIDATION RESULT DATA CLASS
# ============================================================================

@dataclass
class ValidationResult:
    """
    Result of GPT-5.2 validation.
    
    Contains all fields from spec Section 3.8.
    """
    prompt_id: str
    is_one_word_answer_enforced: bool
    best_one_word_answer: str
    top3_one_word_answers: List[str]
    is_X_best: bool
    ambiguity_score: int  # 0-10
    leaks_answer: bool
    naturalness_score: int  # 0-10
    comments: str
    raw_response: Dict = field(default_factory=dict)
    
    # Computed scores
    v_score: int = 0  # V(p) validation score
    
    def compute_v_score(self, target_word: str) -> int:
        """
        Compute V(p) score based on rubric.
        
        Per spec Section 3.7:
        - +40 if the reasoner says X is single most natural one-word completion
        - +30 if no close alternative within top3
        - +20 if prompt enforces one-word answer
        - +10 if fluent (naturalness >= 7)
        - If any hard fail: V(p)=0
        """
        # Hard fails
        if self.leaks_answer:
            self.v_score = 0
            return 0
        if not self.is_one_word_answer_enforced:
            self.v_score = 0
            return 0
        if self.ambiguity_score > CONFIG['validation']['max_ambiguity_score']:
            self.v_score = 0
            return 0
        if self.naturalness_score < CONFIG['validation']['min_naturalness_score']:
            self.v_score = 0
            return 0
        
        score = 0
        
        # +40 if X is single most natural completion
        if self.is_X_best:
            score += 40
        
        # +30 if no close alternative in top3
        target_lower = target_word.lower()
        alternatives = [w.lower() for w in self.top3_one_word_answers if w.lower() != target_lower]
        if len(alternatives) == 0:
            score += 30
        elif len(alternatives) == 1 and self.ambiguity_score <= 2:
            score += 15  # Partial credit
        
        # +20 if one-word answer enforced
        if self.is_one_word_answer_enforced:
            score += 20
        
        # +10 if fluent
        if self.naturalness_score >= 7:
            score += 10
        
        self.v_score = score
        return score

    @staticmethod
    def from_dict(payload: Dict, prompt_id: str, target_word: str) -> "ValidationResult":
        """Rehydrate ValidationResult from a serialized dict."""
        result = ValidationResult(
            prompt_id=prompt_id,
            is_one_word_answer_enforced=payload.get("is_one_word_answer_enforced", False),
            best_one_word_answer=payload.get("best_one_word_answer", ""),
            top3_one_word_answers=payload.get("top3_one_word_answers", []),
            is_X_best=payload.get("is_X_best", False),
            ambiguity_score=int(payload.get("ambiguity_score", 10) or 10),
            leaks_answer=payload.get("leaks_answer", False),
            naturalness_score=int(payload.get("naturalness_score", 0) or 0),
            comments=payload.get("comments", ""),
            raw_response=payload,
        )

        if "v_score" in payload and payload.get("v_score") is not None:
            try:
                result.v_score = int(payload.get("v_score", 0) or 0)
            except (TypeError, ValueError):
                result.v_score = 0
        else:
            result.compute_v_score(target_word)

        return result
    
    def is_accepted(self, target_word: str) -> bool:
        """
        Check if prompt passes acceptance rules.
        
        Per spec Section 3.8:
        - is_X_best == true
        - ambiguity_score <= 3
        - leaks_answer == false
        - is_one_word_answer_enforced == true
        - naturalness_score >= 7
        """
        return (
            self.is_X_best and
            self.ambiguity_score <= CONFIG['validation']['max_ambiguity_score'] and
            not self.leaks_answer and
            self.is_one_word_answer_enforced and
            self.naturalness_score >= CONFIG['validation']['min_naturalness_score'] and
            self.best_one_word_answer.lower() == target_word.lower()
        )


def build_prompt_text(question_text: str) -> str:
    """
    Build the baseline prompt text from a category question.
    """
    question_text = question_text.strip()
    if question_text.startswith("Answer with exactly one English word."):
        return question_text
    return build_prompt(question_text, target_word="", condition="baseline")


# ============================================================================
# GPT-5.2 VALIDATOR
# ============================================================================

class PromptValidator:
    """
    Validate prompts using GPT-5.2.
    
    Sends each prompt to the reasoner for evaluation and parses structured response.
    """
    
    SYSTEM_PROMPT = """You are a rigorous prompt evaluator. Your task is to analyze cloze-style prompts and determine if they have a single, clear best one-word answer.

For each prompt, you must evaluate:
1. Whether the prompt enforces a one-word answer
2. What the best one-word answer is
3. The top 3 plausible one-word answers
4. Whether the target word is the best answer
5. How ambiguous the prompt is (0=perfectly clear, 10=highly ambiguous)
6. Whether the answer is leaked in the prompt
7. How natural/fluent the prompt is (0=awkward, 10=perfectly natural)

Return ONLY valid json in message content. Do not include reasoning or extra text.
Use the literal key name "is_X_best" exactly as written (do not substitute X).
ambiguity_score and naturalness_score must be integers from 0 to 10.
Return your evaluation as a json object with these exact fields:
{
  "is_one_word_answer_enforced": true/false,
  "best_one_word_answer": "word",
  "top3_one_word_answers": ["word1", "word2", "word3"],
  "is_X_best": true/false,
  "ambiguity_score": 0-10,
  "leaks_answer": true/false,
  "naturalness_score": 0-10,
  "comments": "brief explanation"
}"""

    def __init__(self, client: Optional[OpenAIClient] = None):
        self.client = client or OpenAIClient()
        self.validation_log: List[Dict] = []

    def _build_prompt_id(self, prompt: CandidatePrompt) -> str:
        import hashlib
        content = f"{prompt.question_text}||{prompt.target_word}||{prompt.category}"
        hash_full = hashlib.sha256(content.encode('utf-8')).hexdigest()
        hash8 = hash_full[:8]
        return f"{prompt.category}_{prompt.target_word_normalized}_{hash8}"

    def _build_user_prompt(self, prompt: CandidatePrompt) -> str:
        prompt_text = build_prompt_text(prompt.question_text)
        return (
            "Evaluate this cloze prompt:\n\n"
            f"PROMPT:\n{prompt_text}\n\n"
            f"TARGET WORD: {prompt.target_word}\n\n"
            f"Is \"{prompt.target_word}\" the single best one-word answer for this prompt?\n\n"
            "Return your evaluation as json."
        )
    
    def validate_prompt(
        self,
        prompt: CandidatePrompt,
        prompt_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a single prompt using GPT-5.2.
        
        Args:
            prompt: The candidate prompt to validate
            prompt_id: Optional ID for tracking
            
        Returns:
            ValidationResult with parsed scores
        """
        if prompt_id is None:
            prompt_id = self._build_prompt_id(prompt)

        results = self.validate_batch([prompt])
        if results:
            return results[0][1]

        return ValidationResult(
            prompt_id=prompt_id,
            is_one_word_answer_enforced=False,
            best_one_word_answer="",
            top3_one_word_answers=[],
            is_X_best=False,
            ambiguity_score=10,
            leaks_answer=False,
            naturalness_score=0,
            comments="Validation error: empty batch result",
        )
    
    def validate_batch(
        self,
        prompts: List[CandidatePrompt],
        progress_callback=None,
    ) -> List[Tuple[CandidatePrompt, ValidationResult]]:
        """
        Validate a batch of prompts.
        
        Args:
            prompts: List of prompts to validate
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of (prompt, result) tuples
        """
        if not prompts:
            return []

        requests: List[Dict[str, Any]] = []
        prompt_ids: List[str] = []
        for prompt in prompts:
            prompt_id = self._build_prompt_id(prompt)
            prompt_ids.append(prompt_id)
            user_prompt = self._build_user_prompt(prompt)
            requests.append(
                self.client.build_batch_request(
                    custom_id=prompt_id,
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    model=CONFIG.get("openai", {}).get("model", "gpt-5.2-2025-12-11"),
                    temperature=0.1,
                    max_output_tokens=CONFIG.get("openai", {}).get("max_output_tokens_validation", 2000),
                    text_format=CONFIG.get("openai", {}).get("text_format", {"type": "json_object"}),
                )
            )

        try:
            base_paths = get_base_paths()
            batch_dir = base_paths['data_root'] / "batches"
        except Exception:
            batch_dir = Path("batches")

        batch_results = self.client.run_batch_requests(
            requests=requests,
            batch_dir=batch_dir,
            batch_name="validation",
        )

        results: List[Tuple[CandidatePrompt, ValidationResult]] = []
        for i, prompt in enumerate(prompts):
            prompt_id = prompt_ids[i]
            payload = batch_results.get(prompt_id)
            if payload is None or payload.get("error"):
                error_msg = payload.get("error", {}).get("message") if isinstance(payload, dict) else None
                result = ValidationResult(
                    prompt_id=prompt_id,
                    is_one_word_answer_enforced=False,
                    best_one_word_answer="",
                    top3_one_word_answers=[],
                    is_X_best=False,
                    ambiguity_score=10,
                    leaks_answer=False,
                    naturalness_score=0,
                    comments=f"Validation error: {error_msg or 'batch response missing'}",
                )
                results.append((prompt, result))
                continue

            parsed = self.client.extract_json_from_batch_payload(payload)
            if not isinstance(parsed, dict):
                result = ValidationResult(
                    prompt_id=prompt_id,
                    is_one_word_answer_enforced=False,
                    best_one_word_answer="",
                    top3_one_word_answers=[],
                    is_X_best=False,
                    ambiguity_score=10,
                    leaks_answer=False,
                    naturalness_score=0,
                    comments="Validation error: invalid JSON response",
                )
                results.append((prompt, result))
                continue

            result = ValidationResult(
                prompt_id=prompt_id,
                is_one_word_answer_enforced=parsed.get("is_one_word_answer_enforced", False),
                best_one_word_answer=parsed.get("best_one_word_answer", ""),
                top3_one_word_answers=parsed.get("top3_one_word_answers", []),
                is_X_best=parsed.get("is_X_best", False),
                ambiguity_score=int(parsed.get("ambiguity_score", 10)),
                leaks_answer=parsed.get("leaks_answer", False),
                naturalness_score=int(parsed.get("naturalness_score", 0)),
                comments=parsed.get("comments", ""),
                raw_response=parsed,
            )
            result.compute_v_score(prompt.target_word)

            self.validation_log.append({
                "prompt_id": prompt_id,
                "target": prompt.target_word,
                "v_score": result.v_score,
                "accepted": result.is_accepted(prompt.target_word),
            })

            results.append((prompt, result))

            if progress_callback:
                progress_callback(i + 1, len(prompts))

        return results


def compute_semantic_pressure(
    validated_prompts: List['ValidatedPrompt'],
    model_wrapper: Optional[ModelWrapper] = None,
) -> List['ValidatedPrompt']:
    """
    Compute semantic pressure P(p) for validated prompts using the model.

    Uses compute_p_sem_for_prompt from metrics_psem module.
    Results are written into ValidatedPrompt.p_sem and S scores are refreshed.

    Args:
        validated_prompts: List of validated prompts to score
        model_wrapper: Optional pre-loaded ModelWrapper instance

    Returns:
        List of validated prompts with p_sem populated
    """
    wrapper = model_wrapper or ModelWrapper.get_instance()

    # Try to load model; fail hard if unavailable
    try:
        if not wrapper.is_loaded:
            wrapper.load()
    except Exception as e:
        raise RuntimeError(f"Could not load model for pressure computation: {e}") from e

    # Check torch availability
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch unavailable; cannot compute semantic pressure.")

    # Compute P_sem for each prompt (re-raise errors per spec hard-fail behavior)
    failures = []
    for vp in validated_prompts:
        try:
            prompt_text = build_prompt_text(vp.candidate.question_text)
            vp.p_sem = compute_p_sem_for_prompt(
                prompt_text,
                vp.candidate.target_word,
                wrapper.model,
                wrapper.tokenizer,
            )
        except Exception as e:
            logger.error(f"Pressure computation failed for {vp.candidate.target_word}: {e}")
            failures.append({
                "target_word": vp.candidate.target_word,
                "prompt_id": vp.validation.prompt_id,
                "error": str(e),
            })
            # Per spec: P_sem unavailability should halt processing, not silently proceed
            raise RuntimeError(
                f"Pressure computation failed for prompt {vp.validation.prompt_id} "
                f"(target: {vp.candidate.target_word}): {e}"
            ) from e

        vp.compute_s_score()

    return validated_prompts


# ============================================================================
# TARGET WORD TRACKER
# ============================================================================

class TargetTracker:
    """
    Track target word usage across categories.
    
    Enforces the repetition cap (same word in at most 2 categories).
    """
    
    def __init__(self, max_repetition: int = 2):
        self.max_repetition = max_repetition
        self.word_category_map: Dict[str, Set[str]] = {}  # word -> set of categories
    
    def can_use(self, word: str, category: str) -> bool:
        """
        Check if word can be used in this category.
        
        Args:
            word: The target word (normalized)
            category: The category name
            
        Returns:
            True if word can be used
        """
        word_lower = word.lower()
        
        if word_lower not in self.word_category_map:
            return True
        
        categories = self.word_category_map[word_lower]
        
        # Already used in this category is OK
        if category in categories:
            return True
        
        # Check if adding would exceed limit
        return len(categories) < self.max_repetition
    
    def register(self, word: str, category: str) -> bool:
        """
        Register a word as used in a category.
        
        Args:
            word: The target word (normalized)
            category: The category name
            
        Returns:
            True if successfully registered, False if would exceed limit
        """
        word_lower = word.lower()
        
        if not self.can_use(word_lower, category):
            return False
        
        if word_lower not in self.word_category_map:
            self.word_category_map[word_lower] = set()
        
        self.word_category_map[word_lower].add(category)
        return True
    
    def get_usage(self, word: str) -> Set[str]:
        """Get categories where word is used."""
        return self.word_category_map.get(word.lower(), set())
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        multi_use = sum(1 for cats in self.word_category_map.values() if len(cats) > 1)
        return {
            "total_unique_words": len(self.word_category_map),
            "multi_category_words": multi_use,
            "max_categories_per_word": max(len(cats) for cats in self.word_category_map.values()) if self.word_category_map else 0,
        }


# ============================================================================
# SELECTION AND FILTERING
# ============================================================================

@dataclass
class ValidatedPrompt:
    """
    A prompt that has passed validation.
    
    Ready for pressure probing and final selection.
    """
    candidate: CandidatePrompt
    validation: ValidationResult
    prompt_id: str = ""
    v_score: int = 0
    p_sem: float = 0.0  # Semantic pressure (filled by Colab)
    s_score: float = 0.0  # Combined S(p) = V(p) + 100 * P(p)
    
    def compute_s_score(self) -> float:
        """Compute combined selection score."""
        self.s_score = self.v_score + 100 * self.p_sem
        return self.s_score
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        validation_payload = dict(self.validation.raw_response) if self.validation.raw_response else {}
        validation_payload.setdefault("is_one_word_answer_enforced", self.validation.is_one_word_answer_enforced)
        validation_payload.setdefault("best_one_word_answer", self.validation.best_one_word_answer)
        validation_payload.setdefault("top3_one_word_answers", self.validation.top3_one_word_answers)
        validation_payload.setdefault("is_X_best", self.validation.is_X_best)
        validation_payload.setdefault("ambiguity_score", self.validation.ambiguity_score)
        validation_payload.setdefault("leaks_answer", self.validation.leaks_answer)
        validation_payload.setdefault("naturalness_score", self.validation.naturalness_score)
        validation_payload.setdefault("comments", self.validation.comments)
        validation_payload["v_score"] = self.v_score

        prompt_id = self.prompt_id or self.validation.prompt_id
        return {
            "prompt_id": prompt_id,
            **self.candidate.to_dict(),
            "prompt_text": build_prompt_text(self.candidate.question_text),
            "negative_prompt_text": build_prompt(
                self.candidate.question_text,
                self.candidate.target_word,
                "negative",
            ),
            "v_score": self.v_score,
            "p_sem": self.p_sem,
            "s_score": self.s_score,
            "validation": validation_payload,
        }

    @staticmethod
    def from_dict(payload: Dict) -> "ValidatedPrompt":
        """Rehydrate ValidatedPrompt from a serialized dict."""
        candidate = CandidatePrompt.from_dict(payload)
        prompt_id = payload.get("prompt_id", "")
        validation_payload = payload.get("validation", {}) or {}
        validation = ValidationResult.from_dict(
            validation_payload,
            prompt_id=prompt_id,
            target_word=candidate.target_word,
        )

        v_score = payload.get("v_score", validation.v_score)
        try:
            v_score = int(v_score)
        except (TypeError, ValueError):
            v_score = validation.v_score

        p_sem = payload.get("p_sem", 0.0)
        try:
            p_sem = float(p_sem)
        except (TypeError, ValueError):
            p_sem = 0.0

        s_score = payload.get("s_score", 0.0)
        try:
            s_score = float(s_score)
        except (TypeError, ValueError):
            s_score = 0.0

        return ValidatedPrompt(
            candidate=candidate,
            validation=validation,
            prompt_id=prompt_id,
            v_score=v_score,
            p_sem=p_sem,
            s_score=s_score,
        )


class PromptSelector:
    """
    Select best prompts from validated candidates.
    
    Implements:
    - K-per-target selection (max S score)
    - Pressure gating (P >= threshold)
    - Target word repetition control
    - Pressure bin balancing
    """
    
    def __init__(
        self,
        target_tracker: Optional[TargetTracker] = None,
        min_pressure: float = 0.20,
    ):
        self.target_tracker = target_tracker or TargetTracker()
        self.min_pressure = min_pressure
        self.pressure_bins = CONFIG['dataset']['pressure_bins']
        self._pressure_scored = False

    def ensure_pressure_scores(self, prompts: List[ValidatedPrompt]) -> List[ValidatedPrompt]:
        """
        Ensure semantic pressure is populated before gating/balancing.

        Args:
            prompts: Validated prompts (p_sem may be zeroed)

        Returns:
            Prompts with p_sem/s_score filled
        """
        if self._pressure_scored:
            return prompts

        compute_semantic_pressure(prompts)
        self._pressure_scored = True
        return prompts
    
    def select_best_from_candidates(
        self,
        candidates: List[ValidatedPrompt],
    ) -> Optional[ValidatedPrompt]:
        """
        Select best candidate from K candidates for same target.
        
        Uses S(p) = V(p) + 100 * P(p) as selection criterion.
        Tie-breaker: shorter prompt.
        
        Args:
            candidates: List of validated prompts for same target
            
        Returns:
            Best candidate or None if all rejected
        """
        # Filter to accepted only
        valid = [c for c in candidates if c.validation.is_accepted(c.candidate.target_word)]
        
        if not valid:
            return None
        
        # Compute S scores
        for c in valid:
            c.compute_s_score()
        
        # Sort by S score (descending), then by prompt length (ascending)
        valid.sort(key=lambda c: (-c.s_score, len(c.candidate.question_text)))
        
        return valid[0]
    
    def gate_by_pressure(
        self,
        prompts: List[ValidatedPrompt],
        min_pressure: Optional[float] = None,
    ) -> List[ValidatedPrompt]:
        """
        Filter prompts by minimum semantic pressure.
        
        Args:
            prompts: List of validated prompts with P_sem filled
            min_pressure: Minimum pressure threshold
            
        Returns:
            Filtered list of prompts above threshold
        """
        self.ensure_pressure_scores(prompts)
        threshold = min_pressure or self.min_pressure
        return [p for p in prompts if p.p_sem >= threshold]
    
    def balance_by_pressure_bins(
        self,
        prompts: List[ValidatedPrompt],
        target_per_bin: int = 100,
    ) -> List[ValidatedPrompt]:
        """
        Balance prompts across pressure bins.
        
        Per spec Section 3.9:
        - 5 bins: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
        - Aim for target_per_bin prompts per bin
        
        Args:
            prompts: List of validated prompts
            target_per_bin: Target prompts per bin
            
        Returns:
            Balanced list of prompts
        """
        prompts = self.ensure_pressure_scores(prompts)
        bins = {i: [] for i in range(5)}
        
        for p in prompts:
            bin_idx = min(int(p.p_sem * 5), 4)  # 0-4
            bins[bin_idx].append(p)
        
        selected = []
        for bin_idx, bin_prompts in bins.items():
            # Sort by S score and take top
            bin_prompts.sort(key=lambda p: -p.s_score)
            selected.extend(bin_prompts[:target_per_bin])
        
        logger.info(f"Pressure bin distribution: {[len(bins[i]) for i in range(5)]}")
        return selected
    
    def filter_by_target_repetition(
        self,
        prompts: List[ValidatedPrompt],
    ) -> List[ValidatedPrompt]:
        """
        Filter prompts to respect target word repetition limit.
        
        Args:
            prompts: List of validated prompts
            
        Returns:
            Filtered list respecting repetition caps
        """
        selected = []
        
        for p in prompts:
            word = p.candidate.target_word_normalized
            category = p.candidate.category
            
            if self.target_tracker.register(word, category):
                selected.append(p)
            else:
                logger.debug(f"Skipping {word} in {category} - repetition limit reached")
        
        return selected


# ============================================================================
# UNIT TESTS
# ============================================================================


def validate_and_enrich_prompts(
    prompts: List[CandidatePrompt],
    validator: PromptValidator,
    model_wrapper: Optional[ModelWrapper] = None,
    output_file: Optional[Path] = None,
    compute_pressure: bool = True,
    append: bool = False,
    write_output: bool = True,
) -> List[ValidatedPrompt]:
    """
    Validate prompts, compute semantic pressure, and persist enriched results.

    Args:
        prompts: Candidate prompts to validate
        validator: PromptValidator instance
        model_wrapper: Optional preloaded ModelWrapper for scoring
        output_file: Optional path for JSONL export
        compute_pressure: If True, compute P_sem using the model
        append: If True, append to output_file instead of overwriting

    Returns:
        List of ValidatedPrompt objects with v_score, p_sem, and s_score
    """
    filtered_prompts, filter_stats = apply_deterministic_filters(prompts)
    removed = len(prompts) - len(filtered_prompts)
    if removed:
        logger.info(
            "Deterministic filters removed %d prompts (whole_word=%d, stripped=%d, nonalpha=%d, creative_sent=%d, ood_sent=%d).",
            removed,
            filter_stats["whole_word_leak"],
            filter_stats["stripped_leak"],
            filter_stats["nonalpha_target"],
            filter_stats["creative_sentence_count"],
            filter_stats["ood_sentence_count"],
        )

    batch_results = validator.validate_batch(filtered_prompts)
    validated_prompts: List[ValidatedPrompt] = []

    # Prepare gpt5_validation.jsonl path for audit log
    base_paths = None
    if output_file is not None:
        validation_log_path = output_file.parent.parent / "gpt5_validation.jsonl"
    else:
        base_paths = get_base_paths()
        validation_log_path = base_paths['data_root'] / 'gpt5_validation.jsonl'
    validation_log_path.parent.mkdir(parents=True, exist_ok=True)

    for prompt, validation in batch_results:
        vp = ValidatedPrompt(
            candidate=prompt,
            validation=validation,
            prompt_id=validation.prompt_id,
            v_score=validation.v_score,
        )
        validated_prompts.append(vp)
        
        # Write to gpt5_validation.jsonl audit log (append mode)
        audit_record = {
            "prompt_id": validation.prompt_id,
            "category": prompt.category,
            "candidate_text": prompt.question_text,
            "target_word": prompt.target_word,
            "raw_response_json": validation.raw_response,
            "parsed_fields": {
                "is_one_word_answer_enforced": validation.is_one_word_answer_enforced,
                "best_one_word_answer": validation.best_one_word_answer,
                "top3_one_word_answers": validation.top3_one_word_answers,
                "is_X_best": validation.is_X_best,
                "ambiguity_score": validation.ambiguity_score,
                "leaks_answer": validation.leaks_answer,
                "naturalness_score": validation.naturalness_score,
                "comments": validation.comments,
                "v_score": validation.v_score,
            },
            "acceptance_decision": validation.is_accepted(prompt.target_word),
        }
        with open(validation_log_path, 'a', encoding='utf-8') as audit_f:
            json.dump(audit_record, audit_f, ensure_ascii=True)
            audit_f.write('\n')

    if compute_pressure:
        compute_semantic_pressure(validated_prompts, model_wrapper=model_wrapper)

    if write_output:
        # Persist to JSONL for downstream stages
        if output_file is None:
            if base_paths is None:
                base_paths = get_base_paths()
            validated_dir = base_paths['data_root'] / 'validated'
        else:
            validated_dir = output_file.parent
        validated_dir.mkdir(parents=True, exist_ok=True)
        if output_file is None:
            output_file = validated_dir / 'prompts.jsonl'

        write_mode = 'a' if append and output_file.exists() else 'w'
        with open(output_file, write_mode, encoding='utf-8') as f:
            for vp in validated_prompts:
                if vp.s_score == 0:
                    vp.compute_s_score()
                json.dump(vp.to_dict(), f)
                f.write('\n')

        logger.info(f"Saved {len(validated_prompts)} validated prompts to {output_file}")
    logger.info(f"Appended {len(validated_prompts)} records to {validation_log_path}")
    return validated_prompts
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATOR TESTS")
    print("=" * 60)
    
    # Test 1: ValidationResult scoring
    print("\n1. Testing ValidationResult scoring:")
    result = ValidationResult(
        prompt_id="test_1",
        is_one_word_answer_enforced=True,
        best_one_word_answer="paris",
        top3_one_word_answers=["paris", "london", "berlin"],
        is_X_best=True,
        ambiguity_score=2,
        leaks_answer=False,
        naturalness_score=8,
        comments="Good prompt",
    )
    score = result.compute_v_score("paris")
    print(f"   V score for valid prompt: {score}")
    print(f"   Expected: ~70-85")
    print(f"   ✅ Score computed" if score > 0 else "   ❌ Score is 0")
    
    # Test 2: TargetTracker
    print("\n2. Testing TargetTracker:")
    tracker = TargetTracker(max_repetition=2)
    
    # First usage
    r1 = tracker.register("space", "creative")
    print(f"   Register 'space' in creative: {r1}")
    
    # Second category
    r2 = tracker.register("space", "idioms")
    print(f"   Register 'space' in idioms: {r2}")
    
    # Third category should fail
    r3 = tracker.register("space", "facts")
    print(f"   Register 'space' in facts: {r3} (should be False)")
    
    print(f"   ✅ Repetition control works" if not r3 else "   ❌ Repetition control failed")
    
    # Test 3: Mock ValidatedPrompt
    print("\n3. Testing ValidatedPrompt:")
    try:
        from .data_mining import CandidatePrompt
    except ImportError:
        from data_mining import CandidatePrompt
    
    candidate = CandidatePrompt(
        category="facts",
        question_text="The capital of France is ____.",
        target_word="Paris",
        target_word_normalized="paris",
        prompt_style_id="F1",
        source_trace="test",
    )
    
    vp = ValidatedPrompt(
        candidate=candidate,
        validation=result,
        v_score=70,
        p_sem=0.85,
    )
    s = vp.compute_s_score()
    print(f"   S score: {s}")
    print(f"   Expected: 70 + 100*0.85 = 155")
    print(f"   ✅ S score correct" if abs(s - 155) < 0.01 else "   ❌ S score wrong")
    
    print("\n" + "=" * 60)
    print("Validator tests complete!")
    print("=" * 60)
