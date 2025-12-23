"""
data_mining.py - Category-Specific Data Generators

This module provides generators for each prompt category:
- Category A: Idioms (from baiango/english_idioms)
- Category B: Facts (from Wikidata SPARQL)
- Category C: Common Sense (from ConceptNet)
- Category D: Creative (generated via DeepSeek)
- Category E: Out-of-Distribution (generated via DeepSeek)

Each generator produces candidate prompts in a standardized format.
"""

import re
import csv
import json
import random
import logging
from collections import Counter
from io import StringIO
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Import from local modules
try:
    from .api_clients import WikidataClient, ConceptNetClient, DeepSeekClient, download_idioms_csv
    from .config import PROMPT_TEMPLATES, CONFIG, get_base_paths
except ImportError:
    from api_clients import WikidataClient, ConceptNetClient, DeepSeekClient, download_idioms_csv
    from config import PROMPT_TEMPLATES, CONFIG, get_base_paths


# ============================================================================
# CANDIDATE PROMPT DATA CLASS
# ============================================================================

@dataclass
class CandidatePrompt:
    """
    Represents a candidate prompt before validation.
    
    This is the standardized format used across all categories.
    """
    category: str
    question_text: str  # The question/cloze part
    target_word: str    # Expected answer
    target_word_normalized: str  # Lowercase, stripped
    prompt_style_id: str  # Template ID used (e.g., 'I1', 'F2')
    source_trace: str    # Origin info for debugging
    raw_data: Dict = field(default_factory=dict)  # Original data
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category,
            "question_text": self.question_text,
            "target_word": self.target_word,
            "target_word_normalized": self.target_word_normalized,
            "prompt_style_id": self.prompt_style_id,
            "source_trace": self.source_trace,
            "raw_data": self.raw_data,
        }

    @staticmethod
    def from_dict(payload: Dict) -> "CandidatePrompt":
        """Rehydrate CandidatePrompt from a serialized dict."""
        target = payload.get("target_word", "")
        return CandidatePrompt(
            category=payload.get("category", ""),
            question_text=payload.get("question_text", ""),
            target_word=target,
            target_word_normalized=payload.get(
                "target_word_normalized", (target or "").lower()
            ),
            prompt_style_id=payload.get("prompt_style_id", ""),
            source_trace=payload.get("source_trace", ""),
            raw_data=payload.get("raw_data", {}) or {},
        )


# ============================================================================
# TARGET WORD VALIDATION HELPERS
# ============================================================================

def _is_valid_target_word(target: str) -> bool:
    """
    Return True if target is a valid single English word.
    
    Requirements (per issue A):
    - Non-empty after stripping punctuation
    - Strictly alphabetic (target.isalpha())
    - Exactly one token when split on whitespace
    - Length >= 2
    
    Args:
        target: The target word to validate
        
    Returns:
        True if target is valid, False otherwise
    """
    # Strip punctuation from both ends
    stripped = target.strip().strip(".,!?;:\"'-")
    if not stripped:
        return False
    if not stripped.isalpha():
        return False
    if len(stripped.split()) != 1:
        return False
    if len(stripped) < 2:
        return False
    return True


def _target_leaks_into_question(target: str, question: str) -> bool:
    """
    Return True if target appears as a whole word in the question text.
    
    Uses case-insensitive word boundary matching via find_word_occurrences.
    
    Args:
        target: The target word to check for leakage
        question: The question/prompt text to search in
        
    Returns:
        True if target leaks into question, False otherwise
    """
    try:
        from .utils import find_word_occurrences
    except ImportError:
        from utils import find_word_occurrences
    
    occurrences = find_word_occurrences(target, question)
    return len(occurrences) > 0


def _normalize_subject(subject: str) -> str:
    """
    Normalize a common-sense subject to a safe ASCII noun phrase.
    """
    cleaned = subject.strip()
    cleaned = re.sub(r"[^A-Za-z ]", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned


def _infer_common_sense_relation(question: str) -> Optional[str]:
    """
    Best-effort relation inference from a common-sense template.
    """
    if not question:
        return None
    q = question.lower()
    if "made of" in q:
        return "MadeOf"
    if "use a" in q or "use an" in q or "use the" in q:
        if " to " in q:
            return "UsedFor"
    if "typical" in q and " is " in q:
        return "HasProperty"
    return None


def _format_common_sense_question(subject: str, relation: str) -> Optional[Tuple[str, str]]:
    """
    Build a common-sense question from subject + relation.
    """
    relation_map = {
        "UsedFor": ("C1_UsedFor", "Fill the blank (one word): You use a {subject} to ____."),
        "MadeOf": ("C2_MadeOf", "Fill the blank (one word): A {subject} is made of ____."),
        "HasProperty": ("C3_HasProperty", "Fill the blank (one word): A typical {subject} is ____."),
    }
    if relation not in relation_map:
        return None
    style_id, default_template = relation_map[relation]
    template = PROMPT_TEMPLATES.get("common_sense", {}).get(style_id, default_template)
    return style_id, template.format(subject=subject)


def _extract_subject_from_question(question: str, relation: Optional[str]) -> Optional[str]:
    """
    Attempt to recover subject from a common-sense question template.
    """
    if not question:
        return None
    q = question.lower()
    patterns = []
    if relation in (None, "UsedFor"):
        patterns.append(r"use (?:a|an|the) ([a-z ]+) to")
    if relation in (None, "MadeOf"):
        patterns.append(r"(?:a|an|the) ([a-z ]+) is made of")
    if relation in (None, "HasProperty"):
        patterns.append(r"typical ([a-z ]+) is")
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return _normalize_subject(match.group(1))
    return None


def _normalize_fact_relation(relation: str) -> Optional[str]:
    if not relation:
        return None
    rel = relation.strip().lower().replace("-", " ").replace("_", " ")
    if "capital" in rel:
        return "capital"
    if "currency" in rel:
        return "currency"
    return None


def _format_idiom_with_blank(idiom_full: str, target: str) -> Optional[str]:
    if not idiom_full or not target:
        return None
    idiom = idiom_full.strip()
    idiom = idiom.rstrip(".,!?;:\"")
    tokens = idiom.split()
    if len(tokens) < 3:
        return None
    last_token = re.sub(r"[^A-Za-z]", "", tokens[-1])
    if not last_token:
        return None
    if last_token.lower() != target.lower():
        return None
    tokens[-1] = "____"
    return " ".join(tokens)


def _format_creative_microstory(microstory: str) -> Optional[str]:
    if not microstory:
        return None
    cleaned = microstory.strip()
    cleaned = cleaned.rstrip().rstrip(".!?")
    if not cleaned:
        return None
    return cleaned


def _within_zipf_band(word: str) -> bool:
    try:
        from wordfreq import zipf_frequency
    except ImportError as e:
        raise RuntimeError(
            "wordfreq is required to enforce the Zipf band. Install wordfreq to continue."
        ) from e
    min_zipf = CONFIG['wordfreq']['min_zipf']
    max_zipf = CONFIG['wordfreq']['max_zipf']
    score = zipf_frequency(word, 'en')
    return min_zipf <= score <= max_zipf


def _format_ood_prompt_text(
    context: str,
    style: str,
    templates: Dict[str, str],
) -> str:
    template = templates.get(style)
    if not template:
        return f"Fill the blank with one word: {context} ____."

    prefix, sep, rest = template.partition(":")
    if not sep:
        return template

    context = context.strip()
    if context and context[-1] not in ".!?":
        context = context + "."
    rest = rest.strip()

    if context:
        return f"{prefix}: {context} {rest}".strip()
    return f"{prefix}: {rest}".strip()


# ============================================================================
# WORDFREQ TARGET SELECTION
# ============================================================================


def get_wordfreq_targets(
    n: int,
    min_zipf: Optional[float] = None,
    max_zipf: Optional[float] = None,
) -> List[str]:
    """
    Get target words within a Zipf frequency band.

    Falls back to a small built-in list if wordfreq is unavailable.
    """
    min_zipf = CONFIG['wordfreq']['min_zipf'] if min_zipf is None else min_zipf
    max_zipf = CONFIG['wordfreq']['max_zipf'] if max_zipf is None else max_zipf

    try:
        from wordfreq import top_n_list, zipf_frequency
    except ImportError:
        raise RuntimeError(
            "wordfreq is required for target selection. Install wordfreq to continue."
        )

    candidates = top_n_list('en', 5000)
    valid_words = []

    for word in candidates:
        if not word.isalpha():
            continue
        if len(word) < 3:
            continue
        zipf = zipf_frequency(word, 'en')
        if min_zipf <= zipf <= max_zipf:
            valid_words.append(word)
        if len(valid_words) >= n:
            break

    return valid_words


# ============================================================================
# CATEGORY A: IDIOMS
# ============================================================================

class IdiomGenerator:
    """
    Generate prompts from English idioms.
    
    Source: baiango/english_idioms repository
    Format: "{idiom}>>{meaning}"
    """
    
    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize the idiom generator.
        
        Args:
            cache_path: Path to cache downloaded CSV
        """
        self.cache_path = cache_path
        self.idioms: List[Tuple[str, str]] = []  # (idiom, meaning)
        self.templates = PROMPT_TEMPLATES.get('idioms', {})
    
    def load_idioms(self) -> int:
        """
        Load and parse idioms from source.
        
        Returns:
            Number of valid idioms loaded
        """
        # Try cache first
        if self.cache_path and self.cache_path.exists():
            logger.info(f"Loading idioms from cache: {self.cache_path}")
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            logger.info("Downloading idioms from GitHub...")
            content = download_idioms_csv(
                save_path=str(self.cache_path) if self.cache_path else None
            )
        
        self.idioms = []

        def add_idiom(idiom: str, meaning: str = "") -> None:
            idiom = idiom.strip()
            meaning = meaning.strip() if meaning else ""
            if self._is_valid_idiom(idiom):
                self.idioms.append((idiom, meaning))

        rows = list(csv.reader(StringIO(content)))
        if rows:
            header = [h.strip().lower() for h in rows[0]] if rows[0] else []
            idiom_idx = None
            meaning_idx = None
            start_idx = 0

            if header and any("idiom" in h for h in header):
                for i, h in enumerate(header):
                    if "idiom" in h:
                        idiom_idx = i
                        break
                for i, h in enumerate(header):
                    if "meaning" in h or "definition" in h:
                        meaning_idx = i
                        break
                start_idx = 1

            for row in rows[start_idx:]:
                if not row:
                    continue
                if idiom_idx is not None and len(row) > idiom_idx:
                    idiom = row[idiom_idx]
                    meaning = row[meaning_idx] if meaning_idx is not None and len(row) > meaning_idx else ""
                    if idiom:
                        add_idiom(idiom, meaning)
                    continue

                if len(row) == 1:
                    line = row[0].strip()
                    if not line:
                        continue
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    match = re.match(r'\{([^}]+)\}>>\{([^}]+)\}', line)
                    if match:
                        add_idiom(match.group(1), match.group(2))
                elif len(row) >= 2 and idiom_idx is None:
                    idiom = row[0].strip()
                    meaning = row[1].strip() if len(row) > 1 else ""
                    if idiom:
                        add_idiom(idiom, meaning)
        
        logger.info(f"Loaded {len(self.idioms)} valid idioms")
        return len(self.idioms)
    
    def _is_valid_idiom(self, idiom: str) -> bool:
        """
        Check if idiom meets requirements.
        
        Requirements (per spec Section 3.5.A):
        - At least 3 tokens
        - Final token is alphabetic after stripping punctuation
        - No bracket artifacts, repeated underscores, or empty
        """
        # Split into tokens
        tokens = idiom.split()
        if len(tokens) < 3:
            return False
        
        # Get last word and strip punctuation
        last_word = tokens[-1].strip('.,!?;:"\'-')
        if not last_word or not last_word.isalpha():
            return False
        
        # Check for artifacts
        if '[' in idiom or ']' in idiom:
            return False
        if '__' in idiom:
            return False
        
        return True
    
    def _get_target_word(self, idiom: str) -> str:
        """Extract the target word (last word) from idiom."""
        tokens = idiom.split()
        last_word = tokens[-1].strip('.,!?;:"\'-')
        return last_word
    
    def _create_blank_idiom(self, idiom: str) -> str:
        """Replace the last word with blank."""
        tokens = idiom.split()
        # Keep punctuation if present
        last_token = tokens[-1]
        trailing_punct = ''
        for char in reversed(last_token):
            if char in '.,!?;:"\'-':
                trailing_punct = char + trailing_punct
            else:
                break
        
        tokens[-1] = '____' + trailing_punct
        return ' '.join(tokens)
    
    def generate_candidates(self, n: int = 500) -> List[CandidatePrompt]:
        """
        Generate candidate prompts from idioms.
        
        Args:
            n: Number of candidates to generate
            
        Returns:
            List of CandidatePrompt objects
        """
        if not self.idioms:
            self.load_idioms()
        
        # Shuffle and sample
        selected = random.sample(self.idioms, min(n, len(self.idioms)))
        
        candidates = []
        template_ids = list(self.templates.keys())
        
        for idiom, meaning in selected:
            target = self._get_target_word(idiom)
            blank_idiom = self._create_blank_idiom(idiom)
            
            # Pick random template
            style_id = random.choice(template_ids) if template_ids else 'I1'
            template = self.templates.get(style_id, 'Complete the idiom with one word: "{idiom_with_blank}"')
            
            # Format question
            question = template.format(idiom_with_blank=blank_idiom)

            if _target_leaks_into_question(target, question):
                logger.debug("Skipping leaking idiom target: %s in %s", target, question)
                continue
            
            candidates.append(CandidatePrompt(
                category="idioms",
                question_text=question,
                target_word=target,
                target_word_normalized=target.lower(),
                prompt_style_id=style_id,
                source_trace=f"idiom:{idiom}",
                raw_data={"idiom": idiom, "meaning": meaning},
            ))
        
        logger.info(f"Generated {len(candidates)} idiom candidates")
        return candidates


# ============================================================================
# CATEGORY B: FACTS
# ============================================================================

class FactGenerator:
    """
    Generate factual prompts from Wikidata.
    
    Relations:
    - Country -> Capital
    - Country -> Currency
    """
    
    def __init__(self):
        self.client = WikidataClient()
        self.templates = PROMPT_TEMPLATES.get('facts', {})
        self.capitals: List[Tuple[str, str]] = []
        self.currencies: List[Tuple[str, str]] = []
    
    def load_data(self) -> Tuple[int, int]:
        """
        Load data from Wikidata.
        
        Returns:
            Tuple of (num_capitals, num_currencies)
        """
        logger.info("Fetching capitals from Wikidata...")
        self.capitals = self.client.get_capitals()
        
        logger.info("Fetching currencies from Wikidata...")
        self.currencies = self.client.get_currencies()
        
        return len(self.capitals), len(self.currencies)
    
    def generate_candidates(self, n: int = 500) -> List[CandidatePrompt]:
        """
        Generate candidate prompts from facts.
        
        Args:
            n: Number of candidates to generate
            
        Returns:
            List of CandidatePrompt objects
        """
        if not self.capitals and not self.currencies:
            self.load_data()
        
        candidates = []
        
        # Split evenly between capitals and currencies
        n_capitals = n // 2
        n_currencies = n - n_capitals
        
        # Generate capital prompts
        capital_sample = random.sample(
            self.capitals, 
            min(n_capitals, len(self.capitals))
        )
        
        for country, capital in capital_sample:
            # Validate target word (single word, alphabetic, no leakage)
            if not _is_valid_target_word(capital):
                logger.debug(f"Skipping invalid capital target: {capital}")
                continue
            
            # Pick template (F1 or F2 for capitals)
            style_id = random.choice(['F1', 'F2'])
            template = self.templates.get(style_id, 
                'Fill the blank with one word: The capital of {subject} is ____.')
            
            question = template.format(subject=country)
            
            # Check for target leakage into question
            if _target_leaks_into_question(capital, question):
                logger.debug(f"Skipping leaking capital target: {capital} in {question}")
                continue
            
            candidates.append(CandidatePrompt(
                category="facts",
                question_text=question,
                target_word=capital,
                target_word_normalized=capital.lower(),
                prompt_style_id=style_id,
                source_trace=f"wikidata:capital:{country}",
                raw_data={"country": country, "capital": capital},
            ))
        
        # Generate currency prompts
        currency_sample = random.sample(
            self.currencies,
            min(n_currencies, len(self.currencies))
        )
        
        for country, currency in currency_sample:
            # Validate target word (single word, alphabetic, no leakage)
            if not _is_valid_target_word(currency):
                logger.debug(f"Skipping invalid currency target: {currency}")
                continue
            
            style_id = 'F3'
            template = self.templates.get(style_id,
                'Complete with one word: The currency of {subject} is ____.')
            
            question = template.format(subject=country)
            
            # Check for target leakage into question
            if _target_leaks_into_question(currency, question):
                logger.debug(f"Skipping leaking currency target: {currency} in {question}")
                continue
            
            candidates.append(CandidatePrompt(
                category="facts",
                question_text=question,
                target_word=currency,
                target_word_normalized=currency.lower(),
                prompt_style_id=style_id,
                source_trace=f"wikidata:currency:{country}",
                raw_data={"country": country, "currency": currency},
            ))
        
        logger.info(f"Generated {len(candidates)} fact candidates")
        return candidates


# ============================================================================
# CATEGORY C: COMMON SENSE
# ============================================================================

class CommonSenseGenerator:
    """
    Generate common sense prompts from ConceptNet.
    
    Relations:
    - UsedFor: You use a {subject} to ____.
    - MadeOf: A {subject} is made of ____.
    - HasProperty: A typical {subject} is ____.
    """
    
    # Common nouns to query
    DEFAULT_CONCEPTS = [
        "scissors", "hammer", "knife", "pen", "book", "chair", "table", 
        "door", "window", "car", "bicycle", "phone", "computer", "clock",
        "lamp", "bed", "pillow", "blanket", "cup", "plate", "fork", "spoon",
        "umbrella", "coat", "shoe", "hat", "bag", "key", "lock", "mirror",
        "soap", "towel", "brush", "comb", "oven", "fridge", "sink", "toilet",
        "tree", "flower", "grass", "water", "fire", "ice", "snow", "rain",
        "sun", "moon", "star", "cloud", "wind", "stone", "sand", "dirt",
    ]
    
    def __init__(self, concepts: Optional[List[str]] = None):
        self.client = ConceptNetClient()
        self.concepts = concepts or self.DEFAULT_CONCEPTS
        self.templates = PROMPT_TEMPLATES.get('common_sense', {})
        self.relations: Dict[str, List[Tuple[str, str]]] = {}
    
    def load_data(self) -> int:
        """
        Load relations from ConceptNet.
        
        Returns:
            Total number of relations loaded
        """
        logger.info(f"Fetching ConceptNet relations for {len(self.concepts)} concepts...")
        
        self.relations = {
            "UsedFor": [],
            "MadeOf": [],
            "HasProperty": [],
        }
        
        for concept in self.concepts:
            try:
                self.relations["UsedFor"].extend(self.client.get_used_for(concept))
                self.relations["MadeOf"].extend(self.client.get_made_of(concept))
                self.relations["HasProperty"].extend(self.client.get_has_property(concept))
            except Exception as e:
                logger.warning(f"Failed to fetch relations for {concept}: {e}")
        
        total = sum(len(v) for v in self.relations.values())
        logger.info(f"Loaded {total} relations from ConceptNet")
        return total
    
    def generate_candidates(self, n: int = 500) -> List[CandidatePrompt]:
        """
        Generate candidate prompts from common sense relations.
        
        Args:
            n: Number of candidates to generate
            
        Returns:
            List of CandidatePrompt objects
        """
        if not any(self.relations.values()):
            self.load_data()
        
        candidates = []
        n_per_relation = n // 3
        
        relation_map = {
            "UsedFor": ("C1_UsedFor", 
                'Fill the blank (one word): You use a {subject} to ____.'),
            "MadeOf": ("C2_MadeOf",
                'Fill the blank (one word): A {subject} is made of ____.'),
            "HasProperty": ("C3_HasProperty",
                'Fill the blank (one word): A typical {subject} is ____.'),
        }
        
        for relation, (style_id, default_template) in relation_map.items():
            data = self.relations.get(relation, [])
            if not data:
                continue
            
            sample = random.sample(data, min(n_per_relation, len(data)))
            template = self.templates.get(style_id, default_template)
            
            for subject, target in sample:
                # Validate target word (single word, alphabetic, no leakage)
                if not _is_valid_target_word(target):
                    logger.debug(f"Skipping invalid common_sense target: {target}")
                    continue
                
                question = template.format(subject=subject)
                
                # Check for target leakage into question
                if _target_leaks_into_question(target, question):
                    logger.debug(f"Skipping leaking common_sense target: {target} in {question}")
                    continue
                
                candidates.append(CandidatePrompt(
                    category="common_sense",
                    question_text=question,
                    target_word=target,
                    target_word_normalized=target.lower(),
                    prompt_style_id=style_id,
                    source_trace=f"conceptnet:{relation}:{subject}",
                    raw_data={"subject": subject, "relation": relation, "target": target},
                ))
        
        logger.info(f"Generated {len(candidates)} common sense candidates")
        return candidates


# ============================================================================
# CATEGORY D: CREATIVE (DeepSeek Generated)
# ============================================================================

class CreativeGenerator:
    """
    Generate creative micro-story prompts using DeepSeek.
    
    Uses wordfreq to select target words in common frequency band.
    """
    
    SYSTEM_PROMPT = """You are a creative writing assistant. Your task is to generate short micro-stories that have exactly one best one-word completion.

Requirements:
1. Create a 2-sentence micro-story that strongly implies a specific target word
2. Do NOT include the target word anywhere in the micro-story
3. The target word should be the single most natural completion
4. Keep prompts engaging and varied in theme

Format your response as json (a single JSON object). Return json only:
{
  "prompts": [
    {"microstory": "Two sentence story WITHOUT the final blank"}
  ]
}"""

    def __init__(
        self,
        deepseek_client: Optional[DeepSeekClient] = None,
        cache_dir: Optional[Path] = None,
        scenario_texts: Optional[List[str]] = None,
    ):
        self.client = deepseek_client or DeepSeekClient()
        self.cache_dir = cache_dir
        self.scenario_texts = scenario_texts or []
        self.target_words: List[str] = []

    def _get_target_words(self, n: int = 200) -> List[str]:
        """Get target words in the Zipf frequency band."""
        return get_wordfreq_targets(n)

    def _load_scenarios(self, n: int) -> List[str]:
        """
        Load scenario texts from local cache or Hugging Face.

        Returns up to n scenario strings.
        """
        if self.scenario_texts:
            return self.scenario_texts

        search_paths: List[Path] = []
        if self.cache_dir:
            search_paths.extend([
                self.cache_dir / "writingprompts.jsonl",
                self.cache_dir / "writingprompts.txt",
            ])

        try:
            base_paths = get_base_paths()
            search_paths.extend([
                base_paths['data_root'] / "raw" / "writingprompts.jsonl",
                base_paths['data_root'] / "raw" / "writingprompts.txt",
            ])
        except Exception:
            pass

        for path in search_paths:
            if not path or not path.exists():
                continue
            scenarios: List[str] = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if path.suffix == ".jsonl":
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = obj.get("prompt") or obj.get("text") or obj.get("story") or obj.get("context")
                    else:
                        text = line
                    if not text:
                        continue
                    text = text.strip()
                    if len(text.split()) < 6:
                        continue
                    scenarios.append(text)
                    if len(scenarios) >= n:
                        break
            if scenarios:
                self.scenario_texts = scenarios
                return scenarios

        try:
            from datasets import load_dataset
        except ImportError as e:
            logger.warning(
                "datasets not available for WritingPrompts; proceeding without scenario seeds: %s",
                e,
            )
            return []

        scenarios = []
        dataset_name = CONFIG['sources']['writingprompts_dataset']
        try:
            ds = load_dataset(dataset_name, split="train", streaming=True)
            for row in ds:
                text = row.get("prompt") or row.get("text") or row.get("story") or row.get("context")
                if not text:
                    continue
                text = text.strip()
                if len(text.split()) < 6:
                    continue
                scenarios.append(text)
                if len(scenarios) >= n:
                    break
        except Exception as e:
            logger.warning(
                "Failed to stream WritingPrompts dataset; proceeding without scenario seeds: %s",
                e,
            )
            return []

        if scenarios and self.cache_dir:
            cache_path = self.cache_dir / "writingprompts.txt"
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    for s in scenarios:
                        f.write(s.replace('\n', ' ').strip() + '\n')
            except Exception as e:
                logger.debug(f"Could not cache WritingPrompts scenarios: {e}")

        if not scenarios:
            logger.warning(
                "WritingPrompts scenarios unavailable; generating creative prompts without scenario seeds."
            )
            return []

        self.scenario_texts = scenarios
        return scenarios
    
    def generate_candidates(
        self, 
        n: int = 500,
        candidates_per_target: int = 5,
    ) -> List[CandidatePrompt]:
        """
        Generate creative prompt candidates.
        
        Args:
            n: Total candidates to generate
            candidates_per_target: K candidates per target word
            
        Returns:
            List of CandidatePrompt objects
        """
        n_targets = n // candidates_per_target
        self.target_words = self._get_target_words(n_targets)

        candidates = []
        template = PROMPT_TEMPLATES.get('creative', {}).get('default',
            'Fill the blank with one word: {microstory} ____.')

        scenarios = self._load_scenarios(n_targets) if n_targets > 0 else []
        
        for target in self.target_words:
            scenario = random.choice(scenarios) if scenarios else ""
            scenario_line = f"Scenario seed: {scenario}\n" if scenario else ""

            user_prompt = f"""{scenario_line}Generate {candidates_per_target} different micro-story prompts where the answer is "{target}".

Each micro-story should:
- Be 2 sentences long
- NOT contain the word "{target}" anywhere
- Have "{target}" as the single best one-word answer

Return as json with "prompts" array. Each "microstory" should NOT include the blank."""

            try:
                result = self.client.generate_json(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    model=CONFIG.get("deepseek", {}).get("generator_model", "deepseek-reasoner"),
                    temperature=0.7,
                    max_tokens=CONFIG.get("deepseek", {}).get("max_tokens_generation", 2000),
                    required_keys=["prompts"],
                    required_schema={"prompts": list},
                )
                
                prompts = result.get("prompts", [])
                for i, p in enumerate(prompts):
                    if isinstance(p, str):
                        microstory = p.strip()
                    elif isinstance(p, dict):
                        microstory = p.get("microstory", "").strip()
                    else:
                        continue
                    if not microstory:
                        continue
                    if _target_leaks_into_question(target, microstory):
                        continue

                    microstory = microstory.rstrip().rstrip(".!?")
                    question = template.format(microstory=microstory)

                    candidates.append(CandidatePrompt(
                        category="creative",
                        question_text=question,
                        target_word=target,
                        target_word_normalized=target.lower(),
                        prompt_style_id=f"creative_{i}",
                        source_trace=f"deepseek:creative:{target}",
                        raw_data={"microstory": microstory, "target": target},
                    ))
                        
            except Exception as e:
                logger.warning(f"Failed to generate creative prompts for {target}: {e}")
        
        logger.info(f"Generated {len(candidates)} creative candidates")
        return candidates


# ============================================================================
# CATEGORY E: OUT-OF-DISTRIBUTION (DeepSeek Generated)
# ============================================================================

class OODGenerator:
    """
    Generate out-of-distribution prompts using DeepSeek.
    
    Creates unusual, surreal, or pseudo-technical cloze prompts.
    """
    
    SYSTEM_PROMPT = """You are a creative prompt generator specializing in unusual, surreal, and out-of-distribution scenarios.

Your task is to create one-word cloze prompts that are:
1. Unusual or surreal in framing
2. Pseudo-technical, ritualistic, or game-like
3. Still have a single clear best one-word answer
4. Engaging and surprising

The prompts should feel "out of distribution" compared to typical factual or common-sense questions.

Format your response as json (a single JSON object). Return json only:
{
  "prompts": [
    {"context": "1-2 sentence context", "style": "O1"}
  ]
}"""

    def __init__(self, deepseek_client: Optional[DeepSeekClient] = None):
        self.client = deepseek_client or DeepSeekClient()
        self.target_words: List[str] = []

    def _get_target_words(self, n: int = 200) -> List[str]:
        """Get target words in the Zipf frequency band."""
        return get_wordfreq_targets(n)

    def _format_ood_prompt(
        self,
        context: str,
        style: str,
        templates: Dict[str, str],
    ) -> str:
        return _format_ood_prompt_text(context=context, style=style, templates=templates)
    
    def generate_candidates(
        self,
        n: int = 500,
        candidates_per_target: int = 5,
    ) -> List[CandidatePrompt]:
        """
        Generate OOD prompt candidates.
        
        Args:
            n: Number of candidates to generate
            candidates_per_target: K candidates per target word
            
        Returns:
            List of CandidatePrompt objects
        """
        candidates = []
        n_targets = n // candidates_per_target
        self.target_words = self._get_target_words(n_targets)
        
        templates = PROMPT_TEMPLATES.get('ood', {})
        default_style = "O1"
        
        for idx, target in enumerate(self.target_words):
            user_prompt = f"""Generate {candidates_per_target} out-of-distribution cloze contexts where the answer is "{target}".

Each prompt should:
- Present an unusual, surreal, or pseudo-technical scenario
- Make "{target}" the single best one-word completion
- NOT contain the word "{target}" anywhere in the context
- Include 1-2 sentences of context

Choose a style label for each prompt: "O1" or "O2".

Return as json with "prompts" array containing objects with "context" and "style" fields."""

            try:
                result = self.client.generate_json(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    model=CONFIG.get("deepseek", {}).get("generator_model", "deepseek-reasoner"),
                    temperature=0.9,
                    max_tokens=CONFIG.get("deepseek", {}).get("max_tokens_generation", 2000),
                    required_keys=["prompts"],
                    required_schema={"prompts": list},
                )
                
                prompts = result.get("prompts", [])
                for i, p in enumerate(prompts):
                    if isinstance(p, str):
                        context = p.strip()
                        style = default_style
                    elif isinstance(p, dict):
                        context = p.get("context", "").strip()
                        style = p.get("style", default_style) or default_style
                    else:
                        continue
                    
                    if not context:
                        continue
                    if _target_leaks_into_question(target, context):
                        continue

                    question = self._format_ood_prompt(context, style, templates)
                    
                    candidates.append(CandidatePrompt(
                        category="ood",
                        question_text=question,
                        target_word=target,
                        target_word_normalized=target.lower(),
                        prompt_style_id=f"ood_{style}_{idx}_{i}",
                        source_trace=f"deepseek:ood:{target}",
                        raw_data={"context": context, "target": target, "style": style},
                    ))
                        
            except Exception as e:
                logger.warning(f"Failed to generate OOD prompts for {target}: {e}")
        
        logger.info(f"Generated {len(candidates)} OOD candidates")
        return candidates[:n]


# ============================================================================
# DEEPSEEK FALLBACK GENERATOR
# ============================================================================

class DeepSeekFallbackGenerator:
    """
    Generate prompts via DeepSeek when primary sources don't have enough.
    
    Can generate prompts for any category by describing the category style.
    """
    
    CATEGORY_PROMPTS = {
        'idioms': """Generate English idiom completion prompts. Each prompt should:
- Provide the full idiom and the final target_word (last word of the idiom)
- Have exactly ONE correct answer (the original idiom ending)
- Be recognizable and common idioms
- Avoid repeating idioms or target words

Return json: {"prompts": [{"idiom_full": "bite the bullet", "target_word": "bullet"}]}""",

        'facts': """Generate factual one-word answer prompts. Each prompt should:
- Provide a subject (country/entity), a relation ("capital" or "currency"), and a single-word target_word
- Have exactly ONE correct single-word answer
- Be well-known facts that most educated people would know
- Vary domains and avoid repeating targets

Return json: {"prompts": [{"subject": "Japan", "relation": "capital", "target_word": "Tokyo"}]}""",

        'common_sense': """Generate common sense prompts about everyday objects and concepts. Each prompt should:
- Provide a relation label: "UsedFor", "MadeOf", or "HasProperty"
- Provide a simple subject noun (everyday object) and a single-word target_word
- UsedFor -> target_word is a verb (what the subject is used to do)
- MadeOf -> target_word is a material noun
- HasProperty -> target_word is an adjective property
- Avoid repeating subjects or target words; vary domains (kitchen, outdoors, office, school, home)

Return json: {"prompts": [{"subject": "scissors", "relation": "UsedFor", "target_word": "cut"}]}""",

        'creative': """Generate creative micro-story cloze prompts. Each prompt should:
- Provide a 1-2 sentence microstory (without the blank) and a target_word
- The target word should NOT appear in the microstory
- The target word should be the single best one-word completion
- Use common English words (avoid proper nouns or rare terms) for target_word
- Be imaginative and varied in theme
- Avoid repeating themes or target words

Return json: {"prompts": [{"microstory": "The astronaut looked back at Earth, feeling an overwhelming sense of wonder as the stars pulsed quietly.", "target_word": "awe"}]}""",

        'ood': """Generate unusual, out-of-distribution cloze prompts. Each prompt should:
- Provide a 1-2 sentence context, a style label ("O1" or "O2"), and a target_word
- The target word should NOT appear in the context
- Have ONE clear best single-word answer despite the unusual framing
- Feel surprising or unexpected compared to typical questions
- Use common English words (avoid proper nouns or rare terms) for target_word
- Avoid repeating motifs or target words

Return json: {"prompts": [{"context": "According to the ancient ritual, before entering the chamber you must whisper", "style": "O2", "target_word": "silence"}]}""",
    }
    
    SYSTEM_PROMPT = """You are a prompt generator for a research experiment. Generate high-quality cloze-style prompts with single-word answers.

Requirements:
1. Each prompt must have exactly ONE correct single-word answer
2. The answer must be an English word (alphabetic only)
3. The prompt should NOT contain the answer word
4. Prompts should be diverse and natural, with minimal repetition
5. Avoid repeating targets or subjects across prompts

Return your response as valid json only."""

    def __init__(self, client: Optional[DeepSeekClient] = None):
        self.client = client or DeepSeekClient()
    
    def generate_for_category(
        self,
        category: str,
        n: int = 100,
        batch_size: int = 20,
        relation_targets: Optional[Dict[str, int]] = None,
        max_batches: Optional[int] = None,
        max_target_repeats: Optional[int] = None,
        max_subject_repeats: Optional[int] = None,
    ) -> List[CandidatePrompt]:
        """
        Generate prompts for a specific category.
        
        Args:
            category: Category name (idioms, facts, common_sense, creative, ood)
            n: Number of prompts to generate
            batch_size: Prompts per API call
            relation_targets: Optional relation-level target counts (common_sense)
            max_batches: Optional max number of API batches
            max_target_repeats: Optional per-target repetition cap (common_sense)
            max_subject_repeats: Optional per-subject repetition cap (common_sense)
            
        Returns:
            List of CandidatePrompt objects
        """
        if category not in self.CATEGORY_PROMPTS:
            logger.error(f"Unknown category: {category}")
            return []
        
        category_prompt = self.CATEGORY_PROMPTS[category]
        candidates: List[CandidatePrompt] = []
        seen_keys = set()
        target_counts = Counter()
        subject_counts = Counter()
        relation_counts = Counter()
        idiom_templates = PROMPT_TEMPLATES.get('idioms', {})
        fact_templates = PROMPT_TEMPLATES.get('facts', {})
        creative_template = PROMPT_TEMPLATES.get('creative', {}).get(
            'default', 'Fill the blank with one word: {microstory} ____.'
        )
        ood_templates = PROMPT_TEMPLATES.get('ood', {})
        idiom_style_ids = list(idiom_templates.keys()) or ["I1"]
        fact_capital_styles = [s for s in ("F1", "F2") if s in fact_templates] or ["F1"]
        fact_currency_style = "F3" if "F3" in fact_templates else (fact_capital_styles[0])
        idiom_style_idx = 0
        fact_capital_idx = 0

        if category == "common_sense":
            extra_batches = CONFIG['dataset'].get('fallback_max_extra_batches', 0)
            if max_target_repeats is None:
                max_target_repeats = CONFIG['dataset'].get(
                    'common_sense_fallback_max_target_repeats', 1
                )
            if max_subject_repeats is None:
                max_subject_repeats = CONFIG['dataset'].get(
                    'common_sense_fallback_max_subject_repeats', 1
                )
        else:
            extra_batches = CONFIG['dataset'].get('fallback_max_extra_batches', 0)
            relation_targets = None

        target_total = sum(relation_targets.values()) if relation_targets else n
        n_batches = (target_total + batch_size - 1) // batch_size
        max_batches = max_batches or (n_batches + extra_batches)

        batch_idx = 0
        while batch_idx < max_batches:
            remaining = target_total - len(candidates)
            if remaining <= 0:
                break
            
            current_batch = min(batch_size, remaining)
            user_prompt = f"Generate {current_batch} prompts.\n\n{category_prompt}"
            
            try:
                result = self.client.generate_json(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    model=CONFIG.get("deepseek", {}).get("generator_model", "deepseek-reasoner"),
                    temperature=0.7,
                    max_tokens=CONFIG.get("deepseek", {}).get("max_tokens_generation", 2000),
                    required_keys=["prompts"],
                    required_schema={"prompts": list},
                )
                
                prompts = result.get("prompts", [])
                for i, p in enumerate(prompts):
                    if not isinstance(p, dict):
                        continue
                    target = (p.get("target_word", "") or "").strip()
                    if not _is_valid_target_word(target):
                        continue
                    target_norm = target.lower()

                    if category == "common_sense":
                        question = (p.get("question", "") or "").strip()
                        relation = (p.get("relation", "") or "").strip()
                        subject = (p.get("subject", "") or "").strip()
                        if not relation or relation not in ("UsedFor", "MadeOf", "HasProperty"):
                            relation = _infer_common_sense_relation(question)
                        subject = _normalize_subject(subject)
                        if not subject and question:
                            subject = _extract_subject_from_question(question, relation)
                        if not subject or not relation:
                            continue
                        relation_info = _format_common_sense_question(subject, relation)
                        if relation_info is None:
                            continue
                        style_id, question_text = relation_info
                        if relation_targets:
                            target_limit = relation_targets.get(relation, 0)
                            if relation_counts[relation] >= target_limit:
                                continue
                        subject_norm = subject.lower()
                        if max_target_repeats is not None and target_counts[target_norm] >= max_target_repeats:
                            continue
                        if max_subject_repeats is not None and subject_counts[subject_norm] >= max_subject_repeats:
                            continue
                        if _target_leaks_into_question(target, question_text):
                            continue
                        key = (" ".join(question_text.lower().split()), target_norm)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        target_counts[target_norm] += 1
                        subject_counts[subject_norm] += 1
                        relation_counts[relation] += 1
                        candidates.append(CandidatePrompt(
                            category=category,
                            question_text=question_text,
                            target_word=target,
                            target_word_normalized=target_norm,
                            prompt_style_id=style_id,
                            source_trace=f"deepseek:fallback:{category}:batch{batch_idx}",
                            raw_data={
                                "subject": subject,
                                "relation": relation,
                                "target": target,
                            },
                        ))
                        continue

                    if category == "idioms":
                        idiom_full = (p.get("idiom_full", "") or "").strip()
                        idiom_with_blank = _format_idiom_with_blank(idiom_full, target)
                        if not idiom_with_blank:
                            continue
                        style_id = idiom_style_ids[idiom_style_idx % len(idiom_style_ids)]
                        idiom_style_idx += 1
                        template = idiom_templates.get(style_id, 'Fill the blank (one word): {idiom_with_blank}')
                        question_text = template.format(idiom_with_blank=idiom_with_blank)
                        if _target_leaks_into_question(target, question_text):
                            continue
                        key = (" ".join(question_text.lower().split()), target_norm)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        candidates.append(CandidatePrompt(
                            category=category,
                            question_text=question_text,
                            target_word=target,
                            target_word_normalized=target_norm,
                            prompt_style_id=style_id,
                            source_trace=f"deepseek:fallback:{category}:batch{batch_idx}",
                            raw_data={"idiom_full": idiom_full, "idiom_with_blank": idiom_with_blank},
                        ))
                        continue

                    if category == "facts":
                        subject = (p.get("subject", "") or "").strip()
                        relation = _normalize_fact_relation(p.get("relation", "") or "")
                        subject_norm = _normalize_subject(subject)
                        if not subject_norm or relation is None:
                            continue
                        if relation == "capital":
                            style_id = fact_capital_styles[fact_capital_idx % len(fact_capital_styles)]
                            fact_capital_idx += 1
                            template = fact_templates.get(style_id, 'Fill the blank with one word: The capital of {subject} is ____.')
                        else:
                            style_id = fact_currency_style
                            template = fact_templates.get(style_id, 'Complete with one word: The currency of {subject} is ____.')
                        question_text = template.format(subject=subject_norm)
                        if _target_leaks_into_question(target, question_text):
                            continue
                        key = (" ".join(question_text.lower().split()), target_norm)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        candidates.append(CandidatePrompt(
                            category=category,
                            question_text=question_text,
                            target_word=target,
                            target_word_normalized=target_norm,
                            prompt_style_id=style_id,
                            source_trace=f"deepseek:fallback:{category}:batch{batch_idx}",
                            raw_data={"subject": subject_norm, "relation": relation, "target": target},
                        ))
                        continue

                    if category == "creative":
                        microstory = _format_creative_microstory(p.get("microstory", "") or "")
                        if not microstory:
                            continue
                        if not _within_zipf_band(target_norm):
                            continue
                        if _target_leaks_into_question(target, microstory):
                            continue
                        question_text = creative_template.format(microstory=microstory)
                        key = (" ".join(question_text.lower().split()), target_norm)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        candidates.append(CandidatePrompt(
                            category=category,
                            question_text=question_text,
                            target_word=target,
                            target_word_normalized=target_norm,
                            prompt_style_id=f"creative_fallback_{batch_idx}_{i}",
                            source_trace=f"deepseek:fallback:{category}:batch{batch_idx}",
                            raw_data={"microstory": microstory, "target": target},
                        ))
                        continue

                    if category == "ood":
                        context = (p.get("context", "") or "").strip()
                        style = (p.get("style", "") or "O1").strip() or "O1"
                        if style not in ("O1", "O2"):
                            style = "O1"
                        if not context:
                            continue
                        if not _within_zipf_band(target_norm):
                            continue
                        if _target_leaks_into_question(target, context):
                            continue
                        question_text = _format_ood_prompt_text(
                            context=context,
                            style=style,
                            templates=ood_templates,
                        )
                        key = (" ".join(question_text.lower().split()), target_norm)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        candidates.append(CandidatePrompt(
                            category=category,
                            question_text=question_text,
                            target_word=target,
                            target_word_normalized=target_norm,
                            prompt_style_id=f"ood_{style}_fallback_{batch_idx}_{i}",
                            source_trace=f"deepseek:fallback:{category}:batch{batch_idx}",
                            raw_data={"context": context, "style": style, "target": target},
                        ))
                        continue

            except Exception as e:
                logger.warning(f"Fallback generation failed for {category} batch {batch_idx}: {e}")
            
            batch_idx += 1
            if relation_targets:
                met_all = True
                for relation, target_limit in relation_targets.items():
                    if relation_counts[relation] < target_limit:
                        met_all = False
                        break
                if met_all:
                    break

        logger.info(f"Generated {len(candidates)} fallback prompts for {category}")
        return candidates[:target_total]


# ============================================================================
# UNIFIED GENERATOR WITH FALLBACK
# ============================================================================

class DatasetGenerator:
    """
    Unified interface for generating all category candidates.
    
    Automatically uses DeepSeek fallback when primary sources don't provide enough prompts.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        deepseek_client: Optional[DeepSeekClient] = None,
    ):
        self.cache_dir = cache_dir
        self.deepseek_client = deepseek_client or DeepSeekClient()
        
        # Initialize generators
        self.idiom_gen = IdiomGenerator(
            cache_path=cache_dir / "idioms.csv" if cache_dir else None
        )
        self.fact_gen = FactGenerator()
        self.common_sense_gen = CommonSenseGenerator()
        self.creative_gen = CreativeGenerator(self.deepseek_client, cache_dir=cache_dir)
        self.ood_gen = OODGenerator(self.deepseek_client)
        self.fallback_gen = DeepSeekFallbackGenerator(self.deepseek_client)
    
    def generate_all(
        self,
        prompts_per_category: int = 500,
        candidate_multiplier: int = CONFIG['dataset'].get('candidate_multiplier', 2),
        candidates_per_target: Optional[int] = None,
        skip_generated: bool = False,
        use_fallback: bool = True,
    ) -> Dict[str, List[CandidatePrompt]]:
        """
        Generate candidates for all categories.
        
        Automatically uses DeepSeek fallback when categories don't reach target count.
        
        Args:
            prompts_per_category: Target prompts per category
            skip_generated: If True, skip categories requiring DeepSeek (creative, OOD)
            use_fallback: If True, use DeepSeek to fill gaps
            
        Returns:
            Dict mapping category name to list of candidates
        """
        results = {}
        k = candidates_per_target or CONFIG['dataset']['candidates_per_target']
        base_count = max(1, int(prompts_per_category * candidate_multiplier))
        generated_count = max(1, int(prompts_per_category * candidate_multiplier * k))
        desired_counts = {
            'idioms': base_count,
            'facts': base_count,
            'common_sense': base_count,
            'creative': generated_count,
            'ood': generated_count,
        }
        
        # Category A: Idioms
        logger.info("Generating idiom candidates...")
        try:
            results['idioms'] = self.idiom_gen.generate_candidates(desired_counts['idioms'])
        except Exception as e:
            logger.warning("Idiom source failed; proceeding with empty pool: %s", e)
            results['idioms'] = []
        
        # Category B: Facts
        logger.info("Generating fact candidates...")
        try:
            results['facts'] = self.fact_gen.generate_candidates(desired_counts['facts'])
        except Exception as e:
            logger.warning("Facts source failed; proceeding with empty pool: %s", e)
            results['facts'] = []
        
        # Category C: Common Sense
        logger.info("Generating common sense candidates...")
        try:
            results['common_sense'] = self.common_sense_gen.generate_candidates(desired_counts['common_sense'])
        except Exception as e:
            logger.warning("ConceptNet source failed; proceeding with empty pool: %s", e)
            results['common_sense'] = []
        
        if not skip_generated:
            # Category D: Creative
            logger.info("Generating creative candidates...")
            try:
                results['creative'] = self.creative_gen.generate_candidates(
                    desired_counts['creative'],
                    candidates_per_target=k,
                )
            except Exception as e:
                logger.warning("Creative generation failed; proceeding with empty pool: %s", e)
                results['creative'] = []
            
            # Category E: OOD
            logger.info("Generating OOD candidates...")
            try:
                results['ood'] = self.ood_gen.generate_candidates(
                    desired_counts['ood'],
                    candidates_per_target=k,
                )
            except Exception as e:
                logger.warning("OOD generation failed; proceeding with empty pool: %s", e)
                results['ood'] = []
        
        # Fill gaps with fallback if enabled
        if use_fallback:
            for category, candidates in results.items():
                gap = desired_counts.get(category, base_count) - len(candidates)
                if gap <= 0:
                    continue
                logger.info(
                    "Category %s has %d prompts, need %d more. Using DeepSeek fallback...",
                    category,
                    len(candidates),
                    gap,
                )
                if category == "common_sense":
                    relation_types = ["UsedFor", "MadeOf", "HasProperty"]
                    relation_counts = {rel: 0 for rel in relation_types}
                    for cand in candidates:
                        relation = cand.raw_data.get("relation")
                        if not relation and cand.prompt_style_id:
                            if cand.prompt_style_id.startswith("C1_"):
                                relation = "UsedFor"
                            elif cand.prompt_style_id.startswith("C2_"):
                                relation = "MadeOf"
                            elif cand.prompt_style_id.startswith("C3_"):
                                relation = "HasProperty"
                        if relation in relation_counts:
                            relation_counts[relation] += 1

                    relation_targets = {rel: 0 for rel in relation_types}
                    for _ in range(gap):
                        rel = min(relation_types, key=lambda r: relation_counts[r] + relation_targets[r])
                        relation_targets[rel] += 1

                    fallback_prompts = self.fallback_gen.generate_for_category(
                        category,
                        gap,
                        relation_targets=relation_targets,
                    )
                else:
                    fallback_prompts = self.fallback_gen.generate_for_category(category, gap)

                if fallback_prompts:
                    results[category].extend(fallback_prompts)

                # De-duplicate by (question_text, target_word)
                deduped = []
                seen = set()
                for cand in results[category]:
                    key = (" ".join(cand.question_text.lower().split()), cand.target_word_normalized)
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(cand)
                results[category] = deduped
                if len(results[category]) < desired_counts.get(category, base_count):
                    raise RuntimeError(
                        f"Category {category}: only {len(results[category])} candidates after fallback, "
                        f"need {desired_counts.get(category, base_count)}. "
                        "Increase fallback batches or relax filters."
                    )
                logger.info("Category %s now has %d prompts", category, len(results[category]))

        # Final dedupe and diversity logging per category
        for category, candidates in list(results.items()):
            deduped = []
            seen = set()
            for cand in candidates:
                key = (" ".join(cand.question_text.lower().split()), cand.target_word_normalized)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(cand)
            results[category] = deduped
            logger.info(
                "Category %s diversity: %d candidates, %d unique targets, %d unique questions",
                category,
                len(deduped),
                len({c.target_word_normalized for c in deduped}),
                len({" ".join(c.question_text.lower().split()) for c in deduped}),
            )
            if category == "common_sense":
                relation_counts = {"UsedFor": 0, "MadeOf": 0, "HasProperty": 0}
                subject_set = set()
                for cand in deduped:
                    relation = cand.raw_data.get("relation")
                    if relation in relation_counts:
                        relation_counts[relation] += 1
                    subject = cand.raw_data.get("subject")
                    if subject:
                        subject_set.add(" ".join(str(subject).lower().split()))
                logger.info(
                    "Common sense diversity: %d unique subjects, relations=%s",
                    len(subject_set),
                    relation_counts,
                )
        
        # Report final counts
        logger.info("=" * 40)
        logger.info("FINAL PROMPT COUNTS:")
        for cat, prompts in results.items():
            logger.info(f"  {cat}: {len(prompts)}")
        logger.info("=" * 40)
        
        return results


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA MINING TESTS")
    print("=" * 60)
    
    # Test 1: Idioms
    print("\n1. Testing IdiomGenerator:")
    idiom_gen = IdiomGenerator()
    idiom_gen.load_idioms()
    candidates = idiom_gen.generate_candidates(5)
    print(f"   ✅ Generated {len(candidates)} idiom candidates")
    for c in candidates[:2]:
        print(f"      Q: {c.question_text[:50]}...")
        print(f"      A: {c.target_word}")
    
    # Test 2: Facts
    print("\n2. Testing FactGenerator:")
    fact_gen = FactGenerator()
    try:
        candidates = fact_gen.generate_candidates(5)
        print(f"   ✅ Generated {len(candidates)} fact candidates")
        for c in candidates[:2]:
            print(f"      Q: {c.question_text[:50]}...")
            print(f"      A: {c.target_word}")
    except Exception as e:
        print(f"   ⚠️ Skipped (Wikidata may be slow): {e}")
    
    # Test 3: Common Sense (may fail if ConceptNet is down)
    print("\n3. Testing CommonSenseGenerator:")
    cs_gen = CommonSenseGenerator(concepts=["scissors", "hammer"])
    try:
        candidates = cs_gen.generate_candidates(5)
        print(f"   ✅ Generated {len(candidates)} common sense candidates")
    except Exception as e:
        print(f"   ⚠️ Skipped (ConceptNet may be unavailable): {e}")
    
    # Test 4: _is_valid_target_word helper
    print("\n4. Testing _is_valid_target_word:")
    test_cases = [
        ("New York", False),   # Multi-word
        ("ice-cream", False),  # Contains hyphen
        ("Paris", True),       # Valid
        ("", False),           # Empty
        ("a", False),          # Too short
        ("123", False),        # Not alphabetic
        ("Tokyo", True),       # Valid
    ]
    for target, expected in test_cases:
        result = _is_valid_target_word(target)
        status = "✅" if result == expected else "❌"
        print(f"   {status} _is_valid_target_word('{target}') = {result} (expected: {expected})")
    
    # Test 5: _target_leaks_into_question helper
    print("\n5. Testing _target_leaks_into_question:")
    test_cases = [
        ("paris", "The capital of France is Paris.", True),  # Leaks (case-insensitive)
        ("paris", "The capital of France is ____.", False),  # No leak
        ("space", "spacetime is interesting", False),        # Not whole word
        ("space", "I love space travel", True),              # Leaks
    ]
    for target, question, expected in test_cases:
        result = _target_leaks_into_question(target, question)
        status = "✅" if result == expected else "❌"
        print(f"   {status} _target_leaks_into_question('{target}', '{question[:30]}...') = {result}")
    
    print("\n" + "=" * 60)
    print("Data mining tests complete!")
    print("=" * 60)
