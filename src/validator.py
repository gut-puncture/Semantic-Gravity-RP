"""
validator.py - Prompt Validation and Selection

This module provides:
- DeepSeek R1 validation of prompts
- Scoring based on validation rubric
- Pressure probing (requires model - Colab only)
- Selection rule implementation
- Global target word tracking

The "Filter Funnel" as described in the implementation plan.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Import local modules
try:
    from .api_clients import DeepSeekClient
    from .data_mining import CandidatePrompt
    from .config import CONFIG, PROMPT_TEMPLATES, get_base_paths
    from .utils import ModelWrapper
    from .prompt_builder import build_prompt
    from .metrics_psem import compute_p_sem_for_prompt
except ImportError:
    from api_clients import DeepSeekClient
    from data_mining import CandidatePrompt
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
    Result of DeepSeek R1 validation.
    
    Contains all fields from spec Section 3.7.
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
        
        Per spec Section 3.6:
        - +40 if R1 says X is single most natural one-word completion
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
    
    def is_accepted(self, target_word: str) -> bool:
        """
        Check if prompt passes acceptance rules.
        
        Per spec Section 3.7:
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
# DEEPSEEK R1 VALIDATOR
# ============================================================================

class PromptValidator:
    """
    Validate prompts using DeepSeek R1.
    
    Sends each prompt to R1 for evaluation and parses structured response.
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

Return your evaluation as valid JSON with these exact fields:
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

    def __init__(self, client: Optional[DeepSeekClient] = None):
        self.client = client or DeepSeekClient()
        self.validation_log: List[Dict] = []
    
    def validate_prompt(
        self,
        prompt: CandidatePrompt,
        prompt_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a single prompt using DeepSeek R1.
        
        Args:
            prompt: The candidate prompt to validate
            prompt_id: Optional ID for tracking
            
        Returns:
            ValidationResult with parsed scores
        """
        if prompt_id is None:
            prompt_id = f"{prompt.category}_{prompt.target_word_normalized}"
        
        prompt_text = build_prompt_text(prompt.question_text)
        user_prompt = f"""Evaluate this cloze prompt:

PROMPT:
{prompt_text}

TARGET WORD: {prompt.target_word}

Is "{prompt.target_word}" the single best one-word answer for this prompt?

Return your evaluation as JSON."""

        try:
            response = self.client.generate_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model="deepseek-reasoner",  # R1 model
                temperature=0.1,  # Low for consistency
            )
            
            # Parse response
            result = ValidationResult(
                prompt_id=prompt_id,
                is_one_word_answer_enforced=response.get("is_one_word_answer_enforced", False),
                best_one_word_answer=response.get("best_one_word_answer", ""),
                top3_one_word_answers=response.get("top3_one_word_answers", []),
                is_X_best=response.get("is_X_best", False),
                ambiguity_score=int(response.get("ambiguity_score", 10)),
                leaks_answer=response.get("leaks_answer", False),
                naturalness_score=int(response.get("naturalness_score", 0)),
                comments=response.get("comments", ""),
                raw_response=response,
            )
            
            # Compute V score
            result.compute_v_score(prompt.target_word)
            
            # Log
            self.validation_log.append({
                "prompt_id": prompt_id,
                "target": prompt.target_word,
                "v_score": result.v_score,
                "accepted": result.is_accepted(prompt.target_word),
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for {prompt_id}: {e}")
            # Return failed result
            return ValidationResult(
                prompt_id=prompt_id,
                is_one_word_answer_enforced=False,
                best_one_word_answer="",
                top3_one_word_answers=[],
                is_X_best=False,
                ambiguity_score=10,
                leaks_answer=False,
                naturalness_score=0,
                comments=f"Validation error: {e}",
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
        results = []
        
        for i, prompt in enumerate(prompts):
            result = self.validate_prompt(prompt, f"batch_{i}")
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

    # Try to load model; if fails, set all p_sem to 0
    try:
        if not wrapper.is_loaded:
            wrapper.load()
    except Exception as e:
        logger.warning(f"Could not load model for pressure computation: {e}")
        for vp in validated_prompts:
            vp.p_sem = 0.0
            vp.compute_s_score()
        return validated_prompts

    # Check torch availability
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch unavailable; pressure defaults to 0")
        for vp in validated_prompts:
            vp.p_sem = 0.0
            vp.compute_s_score()
        return validated_prompts

    # Compute P_sem for each prompt
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
            vp.p_sem = 0.0

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

        return {
            **self.candidate.to_dict(),
            "prompt_text": build_prompt_text(self.candidate.question_text),
            "v_score": self.v_score,
            "p_sem": self.p_sem,
            "s_score": self.s_score,
            "validation": validation_payload,
        }


class PromptSelector:
    """
    Select best prompts from validated candidates.
    
    Implements:
    - K=5 candidate selection (max S score)
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
        
        Per spec Section 3.8:
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
) -> List[ValidatedPrompt]:
    """
    Validate prompts, compute semantic pressure, and persist enriched results.

    Args:
        prompts: Candidate prompts to validate
        validator: PromptValidator instance
        model_wrapper: Optional preloaded ModelWrapper for scoring
        output_file: Optional path for JSONL export

    Returns:
        List of ValidatedPrompt objects with v_score, p_sem, and s_score
    """
    batch_results = validator.validate_batch(prompts)
    validated_prompts: List[ValidatedPrompt] = []

    for prompt, validation in batch_results:
        vp = ValidatedPrompt(
            candidate=prompt,
            validation=validation,
            v_score=validation.v_score,
        )
        validated_prompts.append(vp)

    compute_semantic_pressure(validated_prompts, model_wrapper=model_wrapper)

    # Persist to JSONL for downstream stages
    base_paths = get_base_paths()
    validated_dir = base_paths['data_root'] / 'validated'
    validated_dir.mkdir(parents=True, exist_ok=True)
    if output_file is None:
        output_file = validated_dir / 'prompts.jsonl'

    with open(output_file, 'w', encoding='utf-8') as f:
        for vp in validated_prompts:
            if vp.s_score == 0:
                vp.compute_s_score()
            json.dump(vp.to_dict(), f)
            f.write('\n')

    logger.info(f"Saved {len(validated_prompts)} validated prompts to {output_file}")
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
