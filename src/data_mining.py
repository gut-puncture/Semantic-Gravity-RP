"""
data_mining.py - Category-Specific Data Generators

This module provides generators for each prompt category:
- Category A: Idioms (from baiango/english_idioms)
- Category B: Facts (from Wikidata SPARQL)
- Category C: Common Sense (from ConceptNet)
- Category D: Creative (generated via GPT-5.2 batch)
- Category E: Out-of-Distribution (generated via GPT-5.2 batch)

Each generator produces candidate prompts in a standardized format.
"""

import re
import csv
import json
import random
import logging
from collections import Counter
from io import StringIO
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Import from local modules
try:
    from .api_clients import WikidataClient, RestCountriesClient, ConceptNetClient, OpenAIClient, download_idioms_csv
    from .config import PROMPT_TEMPLATES, CONFIG, get_base_paths
except ImportError:
    from api_clients import WikidataClient, RestCountriesClient, ConceptNetClient, OpenAIClient, download_idioms_csv
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
    if "used by" in q:
        return "UsedBy"
    if "worn on" in q:
        return "WornOn"
    if "requires" in q:
        return "Requires"
    if "contains" in q:
        return "Contains"
    if "has a" in q:
        return "HasPart"
    if "find a" in q and " in the " in q:
        return "AtLocation"
    if " can " in q:
        return "CapableOf"
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
        "HasPart": ("C4_HasPart", "Fill the blank (one word): A {subject} has a ____."),
        "AtLocation": ("C5_AtLocation", "Fill the blank (one word): You usually find a {subject} in the ____."),
        "CapableOf": ("C6_CapableOf", "Fill the blank (one word): A {subject} can ____."),
        "UsedBy": ("C7_UsedBy", "Fill the blank (one word): A {subject} is used by a ____."),
        "Requires": ("C8_Requires", "Fill the blank (one word): A {subject} requires ____."),
        "Contains": ("C9_Contains", "Fill the blank (one word): A {subject} contains ____."),
        "WornOn": ("C10_WornOn", "Fill the blank (one word): A {subject} is worn on the ____."),
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
    if relation in (None, "HasPart"):
        patterns.append(r"(?:a|an|the) ([a-z ]+) has a")
    if relation in (None, "AtLocation"):
        patterns.append(r"find (?:a|an|the) ([a-z ]+) in the")
    if relation in (None, "CapableOf"):
        patterns.append(r"(?:a|an|the) ([a-z ]+) can")
    if relation in (None, "UsedBy"):
        patterns.append(r"(?:a|an|the) ([a-z ]+) is used by")
    if relation in (None, "Requires"):
        patterns.append(r"(?:a|an|the) ([a-z ]+) requires")
    if relation in (None, "Contains"):
        patterns.append(r"(?:a|an|the) ([a-z ]+) contains")
    if relation in (None, "WornOn"):
        patterns.append(r"(?:a|an|the) ([a-z ]+) is worn on")
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
    if "language" in rel:
        return "language"
    if "continent" in rel:
        return "continent"
    if "mountain" in rel:
        return "highest_mountain"
    if "river" in rel:
        return "longest_river"
    if "lake" in rel:
        return "largest_lake"
    if "animal" in rel:
        return "national_animal"
    if "flower" in rel:
        return "national_flower"
    if "export" in rel:
        return "primary_export"
    if "occupation" in rel or "job" in rel:
        return "occupation"
    if "demonym" in rel or "nationality" in rel:
        return "demonym"
    if "birth month" in rel or "born month" in rel or "birthmonth" in rel:
        return "birth_month"
    if "national sport" in rel or "sport" in rel:
        return "national_sport"
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
    Generate factual prompts from Wikidata + Rest Countries.
    
    Relations:
    - Country -> Capital
    - Country -> Currency
    - Country -> Language
    - Country -> Continent
    - Person -> Occupation
    - Person -> Demonym (citizenship)
    - Person -> Birth month
    - Country -> National sport
    """
    
    def __init__(self):
        self.client = WikidataClient()
        self.rest_client = RestCountriesClient()
        self.templates = PROMPT_TEMPLATES.get('facts', {})
        self.relations: Dict[str, List[Dict[str, str]]] = {}
    
    def load_data(self) -> Dict[str, int]:
        """
        Load data from Wikidata + Rest Countries.
        
        Returns:
            Dict of relation -> count
        """
        self.relations = {
            "capital": [],
            "currency": [],
            "language": [],
            "continent": [],
            "occupation": [],
            "demonym": [],
            "birth_month": [],
            "national_sport": [],
        }
        seen: Dict[str, set] = {key: set() for key in self.relations}

        logger.info("Fetching capitals from Wikidata...")
        try:
            capitals = self.client.get_capitals()
        except Exception as e:
            logger.warning("Wikidata capitals query failed: %s", e)
            capitals = []
        for country, capital in capitals:
            key = (country.lower(), capital.lower())
            if key in seen["capital"]:
                continue
            seen["capital"].add(key)
            self.relations["capital"].append({
                "subject": country,
                "target": capital,
                "source": "wikidata",
            })
        
        logger.info("Fetching currencies from Wikidata...")
        try:
            currencies = self.client.get_currencies()
        except Exception as e:
            logger.warning("Wikidata currencies query failed: %s", e)
            currencies = []
        for country, currency in currencies:
            key = (country.lower(), currency.lower())
            if key in seen["currency"]:
                continue
            seen["currency"].add(key)
            self.relations["currency"].append({
                "subject": country,
                "target": currency,
                "source": "wikidata",
            })

        logger.info("Fetching occupations from Wikidata...")
        try:
            occupations = self.client.get_occupations()
        except Exception as e:
            logger.warning("Wikidata occupations query failed: %s", e)
            occupations = []
        for person, occupation in occupations:
            key = (person.lower(), occupation.lower())
            if key in seen["occupation"]:
                continue
            seen["occupation"].add(key)
            self.relations["occupation"].append({
                "subject": person,
                "target": occupation,
                "source": "wikidata",
            })

        logger.info("Fetching demonyms from Wikidata...")
        try:
            demonyms = self.client.get_demonyms()
        except Exception as e:
            logger.warning("Wikidata demonyms query failed: %s", e)
            demonyms = []
        for person, demonym in demonyms:
            key = (person.lower(), demonym.lower())
            if key in seen["demonym"]:
                continue
            seen["demonym"].add(key)
            self.relations["demonym"].append({
                "subject": person,
                "target": demonym,
                "source": "wikidata",
            })

        logger.info("Fetching birth months from Wikidata...")
        try:
            birth_months = self.client.get_birth_months()
        except Exception as e:
            logger.warning("Wikidata birth month query failed: %s", e)
            birth_months = []
        for person, month in birth_months:
            key = (person.lower(), month.lower())
            if key in seen["birth_month"]:
                continue
            seen["birth_month"].add(key)
            self.relations["birth_month"].append({
                "subject": person,
                "target": month,
                "source": "wikidata",
            })

        logger.info("Fetching national sports from Wikidata...")
        try:
            sports = self.client.get_national_sports()
        except Exception as e:
            logger.warning("Wikidata national sport query failed: %s", e)
            sports = []
        for country, sport in sports:
            key = (country.lower(), sport.lower())
            if key in seen["national_sport"]:
                continue
            seen["national_sport"].add(key)
            self.relations["national_sport"].append({
                "subject": country,
                "target": sport,
                "source": "wikidata",
            })

        try:
            base_paths = get_base_paths()
            cache_path = base_paths["data_root"] / "raw" / "restcountries.json"
        except Exception:
            cache_path = None

        logger.info("Fetching country metadata from Rest Countries...")
        records = self.rest_client.get_country_records(cache_path)
        for record in records:
            name = record.get("name") or ""
            if not name:
                continue

            capitals_rc = [
                c.strip() for c in record.get("capitals", [])
                if isinstance(c, str) and _is_valid_target_word(c.strip())
            ]
            if len(capitals_rc) == 1:
                capital = capitals_rc[0]
                key = (name.lower(), capital.lower())
                if key not in seen["capital"]:
                    seen["capital"].add(key)
                    self.relations["capital"].append({
                        "subject": name,
                        "target": capital,
                        "source": "restcountries",
                    })

            currency_names: List[str] = []
            currencies_rc = record.get("currencies") or {}
            if isinstance(currencies_rc, dict):
                for details in currencies_rc.values():
                    if isinstance(details, dict):
                        currency_name = details.get("name")
                        if isinstance(currency_name, str):
                            currency_names.append(currency_name.strip())
            currency_names = [
                c for c in currency_names if _is_valid_target_word(c)
            ]
            if len(currency_names) == 1:
                currency = currency_names[0]
                key = (name.lower(), currency.lower())
                if key not in seen["currency"]:
                    seen["currency"].add(key)
                    self.relations["currency"].append({
                        "subject": name,
                        "target": currency,
                        "source": "restcountries",
                    })

            languages_rc = record.get("languages") or {}
            language_names = []
            if isinstance(languages_rc, dict):
                for lang in languages_rc.values():
                    if isinstance(lang, str):
                        language_names.append(lang.strip())
            language_names = [
                l for l in language_names if _is_valid_target_word(l)
            ]
            if len(language_names) == 1:
                language = language_names[0]
                key = (name.lower(), language.lower())
                if key not in seen["language"]:
                    seen["language"].add(key)
                    self.relations["language"].append({
                        "subject": name,
                        "target": language,
                        "source": "restcountries",
                    })

            continents_rc = [
                c.strip() for c in record.get("continents", [])
                if isinstance(c, str) and _is_valid_target_word(c.strip())
            ]
            if len(continents_rc) == 1:
                continent = continents_rc[0]
                key = (name.lower(), continent.lower())
                if key not in seen["continent"]:
                    seen["continent"].add(key)
                    self.relations["continent"].append({
                        "subject": name,
                        "target": continent,
                        "source": "restcountries",
                    })

        return {rel: len(items) for rel, items in self.relations.items()}
    
    def generate_candidates(self, n: int = 500) -> List[CandidatePrompt]:
        """
        Generate candidate prompts from facts.
        
        Args:
            n: Number of candidates to generate
            
        Returns:
            List of CandidatePrompt objects
        """
        if not self.relations:
            self.load_data()
        
        candidates = []
        
        relation_templates = {
            "capital": (["F1", "F2"], 'Fill the blank with one word: The capital of {subject} is ____.'),
            "currency": (["F3"], 'Complete with one word: The currency of {subject} is ____.'),
            "language": (["F4"], 'One-word answer: An official language of {subject} is ____.'),
            "continent": (["F5"], 'Fill the blank (one word): {subject} is in ____.'),
            "occupation": (["F12"], 'Fill the blank (one word): A one-word occupation associated with {subject} is ____.'),
            "demonym": (["F13"], 'Fill the blank (one word): A demonym for {subject} is ____.'),
            "birth_month": (["F14"], 'Fill the blank (one word): {subject} was born in the month of ____.'),
            "national_sport": (["F15"], 'Fill the blank (one word): The national sport of {subject} is ____.'),
        }

        available_relations = [rel for rel, items in self.relations.items() if items]
        if not available_relations:
            logger.warning("No fact relations available from Wikidata or Rest Countries.")
            return candidates

        per_relation = n // len(available_relations)
        remainder = n % len(available_relations)

        for idx, relation in enumerate(available_relations):
            items = self.relations.get(relation, [])
            if not items:
                continue
            target_count = per_relation + (1 if idx < remainder else 0)
            sample = random.sample(items, min(target_count, len(items)))
            style_ids, default_template = relation_templates.get(relation, (["F1"], 'Fill the blank with one word: {subject} is ____.'))
            style_idx = 0

            for item in sample:
                country = _normalize_subject(item.get("subject", ""))
                target = (item.get("target", "") or "").strip()
                source = item.get("source", "unknown")
                if not country or not _is_valid_target_word(target):
                    continue
                style_id = style_ids[style_idx % len(style_ids)]
                style_idx += 1
                template = self.templates.get(style_id, default_template)
                question = template.format(subject=country)
                if _target_leaks_into_question(target, question):
                    continue
                candidates.append(CandidatePrompt(
                    category="facts",
                    question_text=question,
                    target_word=target,
                    target_word_normalized=target.lower(),
                    prompt_style_id=style_id,
                    source_trace=f"{source}:{relation}:{country}",
                    raw_data={
                        "subject": country,
                        "relation": relation,
                        "target": target,
                        "source": source,
                    },
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
# CATEGORY D: CREATIVE (GPT-5.2 Generated)
# ============================================================================

class CreativeGenerator:
    """
    Generate creative micro-story prompts using GPT-5.2.
    
    Uses wordfreq to select target words in common frequency band.
    """
    
    SYSTEM_PROMPT = """You are a creative writing assistant. Your task is to generate short micro-stories that have exactly one best one-word completion.

Requirements:
1. Create a 2-sentence micro-story for a cloze test
2. Leave out a word that is strongly implied
2. Do NOT include the target word anywhere in the micro-story
3. The target word should be the single most natural completion
4. Keep prompts engaging and varied in theme
5. Prompts should be very diverse from each other in terms of content, setting, structure, placement of target word etc. 
6. The prompt should simply be a micro-story and should not ask the reader a question, or ask the reader to provide the missing word or even imply the existence of a missing word


Format your response as json (a single JSON object). Return json only:
{
  "prompts": [
    {"microstory": "Two sentence story WITHOUT the final blank"}
  ]
}"""

    def __init__(
        self,
        openai_client: Optional[OpenAIClient] = None,
        cache_dir: Optional[Path] = None,
        scenario_texts: Optional[List[str]] = None,
    ):
        self.client = openai_client or OpenAIClient()
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
        candidates_per_target: int = 3,
    ) -> List[CandidatePrompt]:
        """
        Generate creative prompt candidates.
        
        Args:
            n: Total candidates to generate
            candidates_per_target: K candidates per target word
            
        Returns:
            List of CandidatePrompt objects
        """
        n_targets = (n + candidates_per_target - 1) // candidates_per_target
        self.target_words = self._get_target_words(n_targets)

        candidates = []
        template = PROMPT_TEMPLATES.get('creative', {}).get('default',
            'Fill the blank with one word: {microstory} ____.')

        scenarios = self._load_scenarios(n_targets) if n_targets > 0 else []
        
        batch_requests = []
        batch_meta: Dict[str, Dict[str, str]] = {}
        for target in self.target_words:
            scenario = random.choice(scenarios) if scenarios else ""
            scenario_line = f"Scenario seed: {scenario}\n" if scenario else ""

            user_prompt = f"""{scenario_line}Generate {candidates_per_target} different micro-story prompts where the answer is "{target}".

Each micro-story should:
- Be 2 sentences long
- NOT contain the word "{target}" anywhere
- Have "{target}" as the single best one-word answer

Return as json with "prompts" array. Each "microstory" should NOT include the blank."""

            custom_id = f"creative:{target}"
            batch_requests.append(
                self.client.build_batch_request(
                    custom_id=custom_id,
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    model=CONFIG.get("openai", {}).get("model", "gpt-5.2-2025-12-11"),
                    temperature=0.7,
                    max_output_tokens=CONFIG.get("openai", {}).get("max_output_tokens_generation", 2000),
                    text_format=CONFIG.get("openai", {}).get("text_format", {"type": "json_object"}),
                )
            )
            batch_meta[custom_id] = {"target": target}

        if batch_requests:
            try:
                base_paths = get_base_paths()
                batch_dir = base_paths["data_root"] / "batches"
            except Exception:
                batch_dir = Path("batches")

            batch_results = self.client.run_batch_requests(
                requests=batch_requests,
                batch_dir=batch_dir,
                batch_name="creative_generation",
            )

            for custom_id, payload in batch_results.items():
                meta = batch_meta.get(custom_id)
                if not meta or payload.get("error"):
                    continue
                target = meta["target"]
                parsed = self.client.extract_json_from_batch_payload(payload)
                if not isinstance(parsed, dict):
                    continue
                prompts = parsed.get("prompts", [])
                for i, p in enumerate(prompts):
                    if isinstance(p, str):
                        microstory = p.strip()
                    elif isinstance(p, dict):
                        microstory = (p.get("microstory", "") or "").strip()
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
                        source_trace=f"gpt5:creative:{target}",
                        raw_data={"microstory": microstory, "target": target},
                    ))
        
        logger.info(f"Generated {len(candidates)} creative candidates")
        return candidates


# ============================================================================
# CATEGORY E: OUT-OF-DISTRIBUTION (GPT-5.2 Generated)
# ============================================================================

class OODGenerator:
    """
    Generate out-of-distribution prompts using GPT-5.2.
    
    Creates unusual, surreal, or pseudo-technical cloze prompts.
    """
    
    SYSTEM_PROMPT = """You are a creative prompt generator specializing in unusual, surreal, and out-of-distribution scenarios.

Your task is to create one-word cloze prompts that are:
1. Unusual or surreal in framing
2. Pseudo-technical, ritualistic, or game-like
3. Should be something which would never be written by humans
4. Still have a single clear best one-word answer
5. Engaging and surprising
6. Prompts should be very diverse from each other in terms of content, setting, structure, placement of target word etc. 
7. The prompt should simply be a micro-story and should not ask the reader a question, or ask the reader to provide the missing word or even imply the existence of a missing word


The prompts should feel "out of distribution" compared to typical factual or common-sense questions.

Format your response as json (a single JSON object). Return json only:
{
  "prompts": [
    {"context": "1-2 sentence context", "style": "O1"}
  ]
}"""

    def __init__(self, openai_client: Optional[OpenAIClient] = None):
        self.client = openai_client or OpenAIClient()
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
        candidates_per_target: int = 3,
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
        n_targets = (n + candidates_per_target - 1) // candidates_per_target
        self.target_words = self._get_target_words(n_targets)
        
        templates = PROMPT_TEMPLATES.get('ood', {})
        default_style = "O1"
        
        batch_requests = []
        batch_meta: Dict[str, Dict[str, str]] = {}
        for idx, target in enumerate(self.target_words):
            user_prompt = f"""Generate {candidates_per_target} out-of-distribution cloze contexts where the answer is "{target}".

Each prompt should:
- Present an unusual, surreal, or pseudo-technical scenario
- Make "{target}" the single best one-word completion
- NOT contain the word "{target}" anywhere in the context
- Include 1-2 sentences of context

Choose a style label for each prompt: "O1" or "O2".

Return as json with "prompts" array containing objects with "context" and "style" fields."""

            custom_id = f"ood:{target}:{idx}"
            batch_requests.append(
                self.client.build_batch_request(
                    custom_id=custom_id,
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    model=CONFIG.get("openai", {}).get("model", "gpt-5.2-2025-12-11"),
                    temperature=0.9,
                    max_output_tokens=CONFIG.get("openai", {}).get("max_output_tokens_generation", 2000),
                    text_format=CONFIG.get("openai", {}).get("text_format", {"type": "json_object"}),
                )
            )
            batch_meta[custom_id] = {"target": target, "idx": str(idx)}

        if batch_requests:
            try:
                base_paths = get_base_paths()
                batch_dir = base_paths["data_root"] / "batches"
            except Exception:
                batch_dir = Path("batches")

            batch_results = self.client.run_batch_requests(
                requests=batch_requests,
                batch_dir=batch_dir,
                batch_name="ood_generation",
            )

            for custom_id, payload in batch_results.items():
                meta = batch_meta.get(custom_id)
                if not meta or payload.get("error"):
                    continue
                target = meta["target"]
                idx = int(meta["idx"])
                parsed = self.client.extract_json_from_batch_payload(payload)
                if not isinstance(parsed, dict):
                    continue
                prompts = parsed.get("prompts", [])
                for i, p in enumerate(prompts):
                    if isinstance(p, str):
                        context = p.strip()
                        style = default_style
                    elif isinstance(p, dict):
                        context = (p.get("context", "") or "").strip()
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
                        source_trace=f"gpt5:ood:{target}",
                        raw_data={"context": context, "target": target, "style": style},
                    ))
        
        logger.info(f"Generated {len(candidates)} OOD candidates")
        return candidates[:n]


# ============================================================================
# GPT-5.2 FALLBACK GENERATOR
# ============================================================================

class OpenAIFallbackGenerator:
    """
    Generate prompts via GPT-5.2 when primary sources don't have enough.
    
    Can generate prompts for any category by describing the category style.
    """
    
    CATEGORY_PROMPTS = {
        'idioms': """Generate English idiom completion prompts. Each prompt must follow ALL rules:
1) Output fields:
   - idiom_full: the complete idiom (lowercase, standard spelling, no trailing punctuation)
   - target_word: the last word of idiom_full

2) Uniqueness and correctness:
   - Exactly ONE correct answer exists: target_word must be the canonical final word of the idiom in standard modern English.
   - Avoid idioms with common alternate endings, optional clauses, or extended variants.
   - Avoid idioms where the final word is commonly replaced with a synonym or where multiple spellings are common.
   - Do not repeat idioms or target_word values across prompts.

3) Commonness:
   - Use recognizable, common idioms that most fluent English speakers know.

4) Target word constraints (make the ending non-trivial):
   - target_word must be a content word (noun, main verb, or adjective).
   - target_word must NOT be a function word such as an article, pronoun, auxiliary verb, conjunction, preposition, or verb particle.
   - Disallow target_word if it is in this set:
     {a, an, the, and, or, but, nor, so, yet, to, of, for, with, from, by, at, in, on, into, onto, over, under,
      up, down, out, off, away, back, around, through, as, than, that, this, these, those, who, whom, which, what,
      when, where, why, how, it, its, he, she, they, we, you, i, me, him, her, them, us, my, your, our, their,
      is, are, was, were, be, been, being}
   - Also disallow “light verbs” as target_word:
     {do, does, did, done, have, has, had, get, got, go, went, make, made, take, took, put, keep, kept, let, set}

Return ONLY valid JSON in exactly this shape:
{"prompts":[{"idiom_full":"bite the bullet","target_word":"bullet"}]}
""",

        'facts': """Generate factual one-word answer prompts. Each prompt must follow ALL rules:
1) Output fields:
   - subject: a country or well-known entity (proper noun)
   - relation: one of {capital, currency, language, continent, highest_mountain, longest_river, largest_lake, national_animal, national_flower, primary_export, occupation, demonym, birth_month, national_sport}
   - target_word: the single correct one-word answer

2) One-word answer requirement:
   - target_word must be ONE token with letters only (a–z or A–Z). No spaces. No punctuation. No accents/diacritics.
   - Avoid items where common answers include punctuation or multiword forms (for example “Washington, D.C.” or “Mexico City”).

3) Exactly one correct answer:
   - The fact must be unambiguous and have a single dominant expected answer for the given subject + relation.
   - Avoid disputed cases, multiple official answers, or common alternate spellings.

4) Diversity within this batch:
   - Use all relation types across this batch (do not collapse to only capital/currency).
   - Use countries from diverse regions (Africa, Asia, Europe, Americas, Oceania).
   - Do not repeat subjects or target_word values within this batch.
   - Avoid overusing common currency names (dollar, peso, franc, pound) within this batch.

5) Batch uniqueness:
   - The 20 items in this batch must be mutually distinct in subject, target_word, and relation.
   - If you cannot produce 20 unique items, return fewer rather than repeating.

Examples (one per relation type; match the relation label exactly):
- capital: France -> Paris
- currency: Japan -> Yen
- language: Spain -> Spanish
- continent: Kenya -> Africa
- highest_mountain: Tanzania -> Kilimanjaro
- longest_river: Egypt -> Nile
- largest_lake: Uganda -> Victoria
- national_animal: Australia -> Kangaroo
- national_flower: Netherlands -> Tulip
- primary_export: Saudi Arabia -> Oil
- occupation: Ada Lovelace -> Mathematician
- demonym: Leonardo da Vinci -> Italian
- birth_month: Albert Einstein -> March
- national_sport: Canada -> Hockey

Return ONLY valid JSON in exactly this shape:
{"prompts":[{"subject":"Japan","relation":"capital","target_word":"Tokyo"}]}
""",

        'common_sense': """Generate common sense prompts about everyday objects and concepts. Each prompt must follow ALL rules:
1) Output fields:
   - subject: a simple everyday object noun (optionally a simple compound noun like “paper towel” to remove ambiguity)
   - relation: one of {UsedFor, MadeOf, HasProperty, HasPart, AtLocation, CapableOf, UsedBy, Requires, Contains, WornOn}
   - target_word: a single-word answer

2) Relation semantics:
   - UsedFor: target_word must be a base-form verb describing the single most typical use of the subject.
   - MadeOf: target_word must be a material noun that is the typical/implied material for the subject.
     If the base object is commonly made from many materials (for example “cup”), use a compound subject
     that makes the material obvious (for example “paper cup” -> paper).
   - HasProperty: target_word must be an objective adjective strongly associated with the subject in everyday context.
   - HasPart: target_word must be a concrete noun that is a physical part of the subject.
   - AtLocation: target_word must be a typical location noun where the subject is usually found.
   - CapableOf: target_word must be a base-form verb the subject can typically do.
   - UsedBy: target_word must be a noun describing who typically uses the subject.
   - Requires: target_word must be a noun describing what the subject typically needs to function.
   - Contains: target_word must be a noun describing what the subject typically contains.
   - WornOn: target_word must be a body-part noun describing where the subject is worn.

3) Exactly one correct answer (minimize ambiguity):
   - Prefer subjects where one association is dominant and obvious.
   - Avoid subjects with many equally-plausible uses/materials/properties.

4) Diversity and balance within this batch:
   - Use all relation types across this batch (do not collapse to only 2-3 relations).
   - Cover multiple everyday domains (kitchen tools, clothing, vehicles, rooms, animals, food, outdoors, school, office, sports, health, electronics).
   - Do not repeat subjects or target_word values within this batch.
   - Avoid abstract concepts and proper nouns.

5) Batch uniqueness:
   - The 20 items in this batch must be mutually distinct in subject, target_word, and relation.
   - If you cannot produce 20 unique items, return fewer rather than repeating.

6) One-word requirement:
   - target_word must be one lowercase word with letters only (no spaces, no punctuation).

Examples (one per relation type; match the relation label exactly):
- UsedFor: scissors -> cut
- MadeOf: glass bottle -> glass
- HasProperty: ice -> cold
- HasPart: bicycle -> wheel
- AtLocation: toothbrush -> bathroom
- CapableOf: bird -> fly
- UsedBy: stethoscope -> doctor
- Requires: candle -> oxygen
- Contains: wallet -> cash
- WornOn: ring -> finger

Return ONLY valid JSON in exactly this shape:
{"prompts":[{"subject":"scissors","relation":"UsedFor","target_word":"cut"}]}
""",

        'creative': """You are a creative writing assistant. Your task is to generate short micro-stories that have exactly one best one-word completion.

Requirements:
1. Create a 2-sentence micro-story for a cloze test
2. Leave out a word that is strongly implied
2. Do NOT include the target word anywhere in the micro-story
3. The target word should be the single most natural completion
4. Keep prompts engaging and varied in theme
5. Prompts should be very diverse from each other in terms of content, setting, structure, placement of target word etc. 
6. The prompt should simply be a micro-story and should not ask the reader a question, or ask the reader to provide the missing word or even imply the existence of a missing word


Format your response as json (a single JSON object). Return json only:
{
  "prompts": [
    {"microstory": "Two sentence story WITHOUT the final blank"}
  ]
}""",

        'ood': """You are a creative prompt generator specializing in unusual, surreal, and out-of-distribution scenarios.

Your task is to create one-word cloze prompts that are:
1. Unusual or surreal in framing
2. Pseudo-technical, ritualistic, or game-like
3. Should be something which would never be written by humans
4. Still have a single clear best one-word answer
5. Engaging and surprising
6. Prompts should be very diverse from each other in terms of content, setting, structure, placement of target word etc. 
7. The prompt should simply be a micro-story and should not ask the reader a question, or ask the reader to provide the missing word or even imply the existence of a missing word


The prompts should feel "out of distribution" compared to typical factual or common-sense questions.

Format your response as json (a single JSON object). Return json only:
{
  "prompts": [
    {"context": "1-2 sentence context", "style": "O1"}
  ]
}""",
    }

    COMMON_SENSE_VARIANTS: Dict[str, List[Dict[str, str]]] = {
        "A": [
            {"label": "UsedFor_CleaningAction", "relation": "UsedFor", "rule": "cleaning action verb", "example": "broom -> sweep"},
            {"label": "MadeOf_NaturalMaterial", "relation": "MadeOf", "rule": "natural material noun", "example": "canoe -> wood"},
            {"label": "HasProperty_Texture", "relation": "HasProperty", "rule": "texture adjective", "example": "sandpaper -> rough"},
            {"label": "HasPart_HandleGrip", "relation": "HasPart", "rule": "grip or handle part noun", "example": "suitcase -> handle"},
            {"label": "AtLocation_Room", "relation": "AtLocation", "rule": "room noun", "example": "toothbrush -> bathroom"},
            {"label": "CapableOf_Motion", "relation": "CapableOf", "rule": "motion verb", "example": "frog -> jump"},
            {"label": "UsedBy_Occupation", "relation": "UsedBy", "rule": "occupation noun", "example": "stethoscope -> doctor"},
            {"label": "Requires_PowerSource", "relation": "Requires", "rule": "power source noun", "example": "lamp -> electricity"},
            {"label": "Contains_Liquid", "relation": "Contains", "rule": "liquid noun", "example": "thermos -> coffee"},
            {"label": "WornOn_BodyPart", "relation": "WornOn", "rule": "body part noun", "example": "ring -> finger"},
        ],
        "B": [
            {"label": "UsedFor_CookingAction", "relation": "UsedFor", "rule": "cooking action verb", "example": "pan -> fry"},
            {"label": "MadeOf_SyntheticMaterial", "relation": "MadeOf", "rule": "synthetic material noun", "example": "raincoat -> nylon"},
            {"label": "HasProperty_Temperature", "relation": "HasProperty", "rule": "temperature adjective", "example": "ice -> cold"},
            {"label": "HasPart_LidCover", "relation": "HasPart", "rule": "lid or cover part noun", "example": "jar -> lid"},
            {"label": "AtLocation_Workspace", "relation": "AtLocation", "rule": "workspace place noun", "example": "stapler -> office"},
            {"label": "CapableOf_Sound", "relation": "CapableOf", "rule": "sound verb", "example": "bell -> ring"},
            {"label": "UsedBy_AgeGroup", "relation": "UsedBy", "rule": "age group noun", "example": "rattle -> baby"},
            {"label": "Requires_Fuel", "relation": "Requires", "rule": "fuel noun", "example": "lawnmower -> gasoline"},
            {"label": "Contains_Food", "relation": "Contains", "rule": "food noun", "example": "lunchbox -> sandwich"},
            {"label": "WornOn_UpperBody", "relation": "WornOn", "rule": "upper-body part noun", "example": "scarf -> neck"},
        ],
        "C": [
            {"label": "UsedFor_WritingAction", "relation": "UsedFor", "rule": "writing action verb", "example": "pen -> write"},
            {"label": "MadeOf_RigidMaterial", "relation": "MadeOf", "rule": "rigid material noun", "example": "anvil -> iron"},
            {"label": "HasProperty_Color", "relation": "HasProperty", "rule": "color adjective", "example": "lemon -> yellow"},
            {"label": "HasPart_Fastener", "relation": "HasPart", "rule": "fastener part noun", "example": "jacket -> zipper"},
            {"label": "AtLocation_Outdoors", "relation": "AtLocation", "rule": "outdoor place noun", "example": "bench -> park"},
            {"label": "CapableOf_Light", "relation": "CapableOf", "rule": "light verb", "example": "firefly -> glow"},
            {"label": "UsedBy_Student", "relation": "UsedBy", "rule": "student role noun", "example": "notebook -> student"},
            {"label": "Requires_Water", "relation": "Requires", "rule": "water noun", "example": "plant -> water"},
            {"label": "Contains_Paper", "relation": "Contains", "rule": "paper noun", "example": "folder -> paper"},
            {"label": "WornOn_Wrist", "relation": "WornOn", "rule": "wrist body part noun", "example": "watch -> wrist"},
        ],
        "D": [
            {"label": "UsedFor_RepairAction", "relation": "UsedFor", "rule": "repair action verb", "example": "wrench -> tighten"},
            {"label": "MadeOf_FlexibleMaterial", "relation": "MadeOf", "rule": "flexible material noun", "example": "rubber band -> rubber"},
            {"label": "HasProperty_Size", "relation": "HasProperty", "rule": "size adjective", "example": "elephant -> large"},
            {"label": "HasPart_ButtonSwitch", "relation": "HasPart", "rule": "button or switch part noun", "example": "remote -> button"},
            {"label": "AtLocation_Storage", "relation": "AtLocation", "rule": "storage place noun", "example": "box -> attic"},
            {"label": "CapableOf_Float", "relation": "CapableOf", "rule": "float verb", "example": "leaf -> float"},
            {"label": "UsedBy_Teacher", "relation": "UsedBy", "rule": "teacher noun", "example": "whiteboard -> teacher"},
            {"label": "Requires_Air", "relation": "Requires", "rule": "air noun", "example": "balloon -> air"},
            {"label": "Contains_Tools", "relation": "Contains", "rule": "tools noun", "example": "toolbox -> tools"},
            {"label": "WornOn_Feet", "relation": "WornOn", "rule": "feet body part noun", "example": "boots -> feet"},
        ],
        "E": [
            {"label": "UsedFor_MeasuringAction", "relation": "UsedFor", "rule": "measuring action verb", "example": "ruler -> measure"},
            {"label": "MadeOf_TransparentMaterial", "relation": "MadeOf", "rule": "transparent material noun", "example": "window -> glass"},
            {"label": "HasProperty_Weight", "relation": "HasProperty", "rule": "weight adjective", "example": "rock -> heavy"},
            {"label": "HasPart_ScreenSurface", "relation": "HasPart", "rule": "screen or surface part noun", "example": "tablet -> screen"},
            {"label": "AtLocation_Vehicle", "relation": "AtLocation", "rule": "vehicle noun", "example": "seatbelt -> car"},
            {"label": "CapableOf_Spin", "relation": "CapableOf", "rule": "spin verb", "example": "top -> spin"},
            {"label": "UsedBy_Doctor", "relation": "UsedBy", "rule": "doctor noun", "example": "scalpel -> doctor"},
            {"label": "Requires_Electricity", "relation": "Requires", "rule": "electricity noun", "example": "printer -> electricity"},
            {"label": "Contains_Money", "relation": "Contains", "rule": "money noun", "example": "wallet -> cash"},
            {"label": "WornOn_Waist", "relation": "WornOn", "rule": "waist body part noun", "example": "belt -> waist"},
        ],
        "F": [
            {"label": "UsedFor_StorageAction", "relation": "UsedFor", "rule": "storage action verb", "example": "bin -> store"},
            {"label": "MadeOf_SoftMaterial", "relation": "MadeOf", "rule": "soft material noun", "example": "pillow -> cotton"},
            {"label": "HasProperty_Sharpness", "relation": "HasProperty", "rule": "sharpness adjective", "example": "knife -> sharp"},
            {"label": "HasPart_WheelSupport", "relation": "HasPart", "rule": "wheel part noun", "example": "cart -> wheel"},
            {"label": "AtLocation_Building", "relation": "AtLocation", "rule": "building place noun", "example": "elevator -> lobby"},
            {"label": "CapableOf_Roll", "relation": "CapableOf", "rule": "roll verb", "example": "ball -> roll"},
            {"label": "UsedBy_Chef", "relation": "UsedBy", "rule": "chef noun", "example": "whisk -> chef"},
            {"label": "Requires_Battery", "relation": "Requires", "rule": "battery noun", "example": "flashlight -> battery"},
            {"label": "Contains_Clothing", "relation": "Contains", "rule": "clothing noun", "example": "suitcase -> clothes"},
            {"label": "WornOn_Hands", "relation": "WornOn", "rule": "hands body part noun", "example": "gloves -> hands"},
        ],
        "G": [
            {"label": "UsedFor_ProtectionAction", "relation": "UsedFor", "rule": "protection action verb", "example": "helmet -> protect"},
            {"label": "MadeOf_HardMaterial", "relation": "MadeOf", "rule": "hard material noun", "example": "statue -> stone"},
            {"label": "HasProperty_Softness", "relation": "HasProperty", "rule": "softness adjective", "example": "blanket -> soft"},
            {"label": "HasPart_BladeEdge", "relation": "HasPart", "rule": "blade or edge part noun", "example": "axe -> blade"},
            {"label": "AtLocation_Kitchen", "relation": "AtLocation", "rule": "kitchen place noun", "example": "spoon -> kitchen"},
            {"label": "CapableOf_OpenClose", "relation": "CapableOf", "rule": "open or close verb", "example": "door -> open"},
            {"label": "UsedBy_Artist", "relation": "UsedBy", "rule": "artist noun", "example": "easel -> artist"},
            {"label": "Requires_Heat", "relation": "Requires", "rule": "heat noun", "example": "oven -> heat"},
            {"label": "Contains_Medicine", "relation": "Contains", "rule": "medicine noun", "example": "pillbox -> pills"},
            {"label": "WornOn_Head", "relation": "WornOn", "rule": "head body part noun", "example": "hat -> head"},
        ],
        "H": [
            {"label": "UsedFor_CommunicationAction", "relation": "UsedFor", "rule": "communication action verb", "example": "phone -> call"},
            {"label": "MadeOf_LightMaterial", "relation": "MadeOf", "rule": "light material noun", "example": "kite -> foam"},
            {"label": "HasProperty_Transparency", "relation": "HasProperty", "rule": "transparency adjective", "example": "glass -> transparent"},
            {"label": "HasPart_StrapCord", "relation": "HasPart", "rule": "strap or cord part noun", "example": "backpack -> strap"},
            {"label": "AtLocation_Bathroom", "relation": "AtLocation", "rule": "bathroom place noun", "example": "soap -> bathroom"},
            {"label": "CapableOf_Bounce", "relation": "CapableOf", "rule": "bounce verb", "example": "ball -> bounce"},
            {"label": "UsedBy_Athlete", "relation": "UsedBy", "rule": "athlete noun", "example": "cleats -> athlete"},
            {"label": "Requires_Signal", "relation": "Requires", "rule": "signal noun", "example": "radio -> signal"},
            {"label": "Contains_Ink", "relation": "Contains", "rule": "ink noun", "example": "pen -> ink"},
            {"label": "WornOn_Neck", "relation": "WornOn", "rule": "neck body part noun", "example": "necklace -> neck"},
        ],
        "I": [
            {"label": "UsedFor_TransportAction", "relation": "UsedFor", "rule": "transport action verb", "example": "truck -> haul"},
            {"label": "MadeOf_DurableMaterial", "relation": "MadeOf", "rule": "durable material noun", "example": "shield -> steel"},
            {"label": "HasProperty_Brightness", "relation": "HasProperty", "rule": "brightness adjective", "example": "sun -> bright"},
            {"label": "HasPart_Hinge", "relation": "HasPart", "rule": "hinge part noun", "example": "door -> hinge"},
            {"label": "AtLocation_School", "relation": "AtLocation", "rule": "school place noun", "example": "locker -> school"},
            {"label": "CapableOf_Grow", "relation": "CapableOf", "rule": "grow verb", "example": "tree -> grow"},
            {"label": "UsedBy_Driver", "relation": "UsedBy", "rule": "driver noun", "example": "steering wheel -> driver"},
            {"label": "Requires_Maintenance", "relation": "Requires", "rule": "maintenance noun", "example": "bike -> maintenance"},
            {"label": "Contains_Seeds", "relation": "Contains", "rule": "seeds noun", "example": "apple -> seeds"},
            {"label": "WornOn_Back", "relation": "WornOn", "rule": "back body part noun", "example": "backpack -> back"},
        ],
        "J": [
            {"label": "UsedFor_CreativeAction", "relation": "UsedFor", "rule": "creative action verb", "example": "brush -> paint"},
            {"label": "MadeOf_ConductiveMaterial", "relation": "MadeOf", "rule": "conductive material noun", "example": "wire -> copper"},
            {"label": "HasProperty_NoiseLevel", "relation": "HasProperty", "rule": "noise-level adjective", "example": "drum -> loud"},
            {"label": "HasPart_NozzleSpout", "relation": "HasPart", "rule": "nozzle or spout part noun", "example": "teapot -> spout"},
            {"label": "AtLocation_Office", "relation": "AtLocation", "rule": "office place noun", "example": "printer -> office"},
            {"label": "CapableOf_Melt", "relation": "CapableOf", "rule": "melt verb", "example": "ice -> melt"},
            {"label": "UsedBy_Parent", "relation": "UsedBy", "rule": "parent noun", "example": "stroller -> parent"},
            {"label": "Requires_Tool", "relation": "Requires", "rule": "tool noun", "example": "screw -> screwdriver"},
            {"label": "Contains_Waste", "relation": "Contains", "rule": "waste noun", "example": "trashcan -> waste"},
            {"label": "WornOn_Legs", "relation": "WornOn", "rule": "legs body part noun", "example": "pants -> legs"},
        ],
    }

    def _build_common_sense_prompt(self, variant_id: str) -> str:
        items = self.COMMON_SENSE_VARIANTS.get(variant_id)
        if not items:
            return self.CATEGORY_PROMPTS["common_sense"]
        labels = ", ".join([item["label"] for item in items])
        example_label = items[0]["label"]
        example_relation = items[0]["relation"]
        example_pair = items[0]["example"].split("->", 1)
        example_subject = example_pair[0].strip() if example_pair else "broom"
        example_target = example_pair[1].strip() if len(example_pair) > 1 else "sweep"
        lines = [
            "Generate common sense prompts about everyday objects and concepts. Each prompt must follow ALL rules:",
            "1) Output fields:",
            "   - subject: a simple everyday object noun (use a compound noun if needed to remove ambiguity)",
            "   - relation: one of {UsedFor, MadeOf, HasProperty, HasPart, AtLocation, CapableOf, UsedBy, Requires, Contains, WornOn}",
            "   - relation_label: one of the labels listed below (use each label exactly once)",
            "   - target_word: a single-word answer (letters only, no spaces, no punctuation)",
            "2) One-word answer requirement:",
            "   - target_word must be one lowercase word with letters only (no spaces, no punctuation).",
            "3) Exactly one correct answer (minimize ambiguity):",
            "   - Prefer subjects where one association is dominant and obvious.",
            "   - Avoid subjects with many equally plausible answers.",
            "4) Diversity within this batch:",
            "   - Use each relation_label exactly once (10 items total).",
            "   - Do not repeat subjects or target_word values within this batch.",
            "   - Avoid abstract concepts and proper nouns.",
            "5) Leakage avoidance:",
            "   - Do NOT include the target_word in the subject or the prompt text.",
            "",
            f"Relation labels for this prompt (use each exactly once): {labels}",
            "Examples for each relation_label (match the relation_label exactly):",
        ]
        for item in items:
            lines.append(
                f"- {item['label']} (relation={item['relation']}): {item['rule']}. Example: {item['example']}"
            )
        lines.extend([
            "",
            "Return ONLY valid JSON in exactly this shape:",
            f"{{\"prompts\":[{{\"subject\":\"{example_subject}\",\"relation\":\"{example_relation}\",\"relation_label\":\"{example_label}\",\"target_word\":\"{example_target}\"}}]}}",
        ])
        return "\n".join(lines)
    
    SYSTEM_PROMPT = """You are a prompt generator for a research experiment. Generate high-quality cloze-style prompts with single-word answers.

Requirements:
1. Each prompt must have exactly ONE correct single-word answer
2. The answer must be an English word (alphabetic only)
3. The prompt should NOT contain the answer word
4. Prompts should be diverse and natural, with minimal repetition
5. Avoid repeating targets or subjects across prompts

Return your response as valid json only."""

    def __init__(self, client: Optional[OpenAIClient] = None):
        self.client = client or OpenAIClient()
    
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
        fact_relation_styles = {
            "capital": (fact_capital_styles, 'Fill the blank with one word: The capital of {subject} is ____.'),
            "currency": ([fact_currency_style], 'Complete with one word: The currency of {subject} is ____.'),
            "language": (["F4"], 'One-word answer: An official language of {subject} is ____.'),
            "continent": (["F5"], 'Fill the blank (one word): {subject} is in ____.'),
            "highest_mountain": (["F6"], 'Fill the blank (one word): The highest mountain in {subject} is ____.'),
            "longest_river": (["F7"], 'Fill the blank (one word): The longest river in {subject} is ____.'),
            "largest_lake": (["F8"], 'Fill the blank (one word): The largest lake in {subject} is ____.'),
            "national_animal": (["F9"], 'Fill the blank (one word): A national animal of {subject} is ____.'),
            "national_flower": (["F10"], 'Fill the blank (one word): A national flower of {subject} is ____.'),
            "primary_export": (["F11"], 'Fill the blank (one word): A primary export of {subject} is ____.'),
            "occupation": (["F12"], 'Fill the blank (one word): A one-word occupation associated with {subject} is ____.'),
            "demonym": (["F13"], 'Fill the blank (one word): A demonym for {subject} is ____.'),
            "birth_month": (["F14"], 'Fill the blank (one word): {subject} was born in the month of ____.'),
            "national_sport": (["F15"], 'Fill the blank (one word): The national sport of {subject} is ____.'),
        }
        fact_relation_counts = Counter()
        common_sense_variant_ids = list(self.COMMON_SENSE_VARIANTS.keys())
        common_sense_label_set = {
            item["label"] for items in self.COMMON_SENSE_VARIANTS.values() for item in items
        }

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
            if batch_size != 10:
                batch_size = 10
        else:
            extra_batches = CONFIG['dataset'].get('fallback_max_extra_batches', 0)
            relation_targets = None

        target_total = sum(relation_targets.values()) if relation_targets else n
        n_batches = (target_total + batch_size - 1) // batch_size
        max_batches = max_batches or (n_batches + extra_batches)

        batch_requests: List[Dict[str, Any]] = []
        for batch_idx in range(max_batches):
            remaining = target_total - (batch_idx * batch_size)
            if remaining <= 0:
                break
            current_batch = min(batch_size, remaining)
            if category == "common_sense" and common_sense_variant_ids:
                variant_id = common_sense_variant_ids[batch_idx % len(common_sense_variant_ids)]
                category_prompt = self._build_common_sense_prompt(variant_id)
                custom_id = f"fallback:{category}:{variant_id}:{batch_idx}"
            else:
                custom_id = f"fallback:{category}:{batch_idx}"
            user_prompt = f"Generate {current_batch} prompts.\n\n{category_prompt}"
            batch_requests.append(
                self.client.build_batch_request(
                    custom_id=custom_id,
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    model=CONFIG.get("openai", {}).get("model", "gpt-5.2-2025-12-11"),
                    temperature=0.7,
                    max_output_tokens=CONFIG.get("openai", {}).get("max_output_tokens_generation", 2000),
                    text_format=CONFIG.get("openai", {}).get("text_format", {"type": "json_object"}),
                )
            )

        if not batch_requests:
            return []

        try:
            base_paths = get_base_paths()
            batch_dir = base_paths["data_root"] / "batches"
        except Exception:
            batch_dir = Path("batches")

        batch_results = self.client.run_batch_requests(
            requests=batch_requests,
            batch_dir=batch_dir,
            batch_name=f"fallback_{category}",
        )

        for batch_idx in range(max_batches):
            if category == "common_sense" and common_sense_variant_ids:
                variant_id = common_sense_variant_ids[batch_idx % len(common_sense_variant_ids)]
                custom_id = f"fallback:{category}:{variant_id}:{batch_idx}"
            else:
                custom_id = f"fallback:{category}:{batch_idx}"
            payload = batch_results.get(custom_id)
            if payload is None or payload.get("error"):
                continue
            parsed = self.client.extract_json_from_batch_payload(payload)
            if not isinstance(parsed, dict):
                continue
            prompts = parsed.get("prompts", [])
            for i, p in enumerate(prompts):
                if not isinstance(p, dict):
                    continue
                target = (p.get("target_word", "") or "").strip()
                if not _is_valid_target_word(target):
                    continue
                target_norm = target.lower()

                if category == "common_sense":
                    question = (p.get("question_text", "") or p.get("question", "") or "").strip()
                    relation = (p.get("relation", "") or "").strip()
                    relation_label = (p.get("relation_label", "") or "").strip()
                    subject = (p.get("subject", "") or "").strip()
                    allowed_relations = (
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
                    )
                    if relation_label:
                        if relation_label not in common_sense_label_set:
                            relation_label = ""
                        else:
                            prefix = relation_label.split("_", 1)[0]
                            if prefix in allowed_relations:
                                relation = prefix
                    if not relation or relation not in allowed_relations:
                        relation = _infer_common_sense_relation(question)
                    subject = _normalize_subject(subject)
                    if not subject and question:
                        subject = _extract_subject_from_question(question, relation)
                    if not subject or not relation:
                        continue
                    if question:
                        question_text = question
                        style_id = f"common_sense_{relation}"
                    else:
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
                        source_trace=f"gpt5:fallback:{category}:batch{batch_idx}",
                        raw_data={
                            "subject": subject,
                            "relation": relation,
                            "relation_label": relation_label,
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
                        source_trace=f"gpt5:fallback:{category}:batch{batch_idx}",
                        raw_data={"idiom_full": idiom_full, "idiom_with_blank": idiom_with_blank},
                    ))
                    continue

                if category == "facts":
                    subject = (p.get("subject", "") or "").strip()
                    relation = _normalize_fact_relation(p.get("relation", "") or "")
                    subject_norm = _normalize_subject(subject)
                    if not subject_norm or relation is None:
                        continue
                    relation_info = fact_relation_styles.get(relation)
                    if not relation_info:
                        continue
                    style_ids, default_template = relation_info
                    style_idx = fact_relation_counts[relation] % len(style_ids)
                    style_id = style_ids[style_idx]
                    fact_relation_counts[relation] += 1
                    template = fact_templates.get(style_id, default_template)
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
                        source_trace=f"gpt5:fallback:{category}:batch{batch_idx}",
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
                        source_trace=f"gpt5:fallback:{category}:batch{batch_idx}",
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
                        source_trace=f"gpt5:fallback:{category}:batch{batch_idx}",
                        raw_data={"context": context, "style": style, "target": target},
                    ))
                    continue

        logger.info(f"Generated {len(candidates)} fallback prompts for {category}")
        return candidates[:target_total]


# ============================================================================
# UNIFIED GENERATOR WITH FALLBACK
# ============================================================================

class DatasetGenerator:
    """
    Unified interface for generating all category candidates.
    
    Automatically uses GPT-5.2 batch fallback when primary sources don't provide enough prompts.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        openai_client: Optional[OpenAIClient] = None,
    ):
        self.cache_dir = cache_dir
        self.openai_client = openai_client or OpenAIClient()
        
        # Initialize generators
        self.idiom_gen = IdiomGenerator(
            cache_path=cache_dir / "idioms.csv" if cache_dir else None
        )
        self.fact_gen = FactGenerator()
        self.common_sense_gen = CommonSenseGenerator()
        self.creative_gen = CreativeGenerator(self.openai_client, cache_dir=cache_dir)
        self.ood_gen = OODGenerator(self.openai_client)
        self.fallback_gen = OpenAIFallbackGenerator(self.openai_client)
    
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
        
        Automatically uses GPT-5.2 batch fallback when categories don't reach target count.
        
        Args:
            prompts_per_category: Target prompts per category
            skip_generated: If True, skip categories requiring GPT-5.2 (creative, OOD)
            use_fallback: If True, use GPT-5.2 to fill gaps
            
        Returns:
            Dict mapping category name to list of candidates
        """
        results = {}
        k = candidates_per_target or CONFIG['dataset']['candidates_per_target']
        base_count = max(1, int(prompts_per_category * candidate_multiplier))
        creative_ood_targets = CONFIG['dataset'].get('creative_ood_target_count', prompts_per_category)
        creative_ood_count = max(1, int(creative_ood_targets * k))
        desired_counts = {
            'idioms': base_count,
            'facts': base_count,
            'common_sense': base_count,
            'creative': creative_ood_count,
            'ood': creative_ood_count,
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
                    "Category %s has %d prompts, need %d more. Using GPT-5.2 batch fallback...",
                    category,
                    len(candidates),
                    gap,
                )
                if category == "common_sense":
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
                    logger.warning(
                        "Category %s: only %d candidates after fallback (target %d). Proceeding without regeneration.",
                        category,
                        len(results[category]),
                        desired_counts.get(category, base_count),
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
                relation_counts = {
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
