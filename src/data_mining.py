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
import random
import logging
from io import StringIO
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Import from local modules
try:
    from .api_clients import WikidataClient, ConceptNetClient, DeepSeekClient, download_idioms_csv
    from .config import PROMPT_TEMPLATES, CONFIG
except ImportError:
    from api_clients import WikidataClient, ConceptNetClient, DeepSeekClient, download_idioms_csv
    from config import PROMPT_TEMPLATES, CONFIG


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
        }


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
        
        # Parse the format: "{idiom}>>{meaning}" (with outer quotes)
        self.idioms = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line or '>>' not in line:
                continue
            
            # Remove outer quotes if present
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            # Parse format: {Break a leg}>>{Good luck!}
            match = re.match(r'\{([^}]+)\}>>\{([^}]+)\}', line)
            if match:
                idiom = match.group(1).strip()
                meaning = match.group(2).strip()
                
                # Validate idiom
                if self._is_valid_idiom(idiom):
                    self.idioms.append((idiom, meaning))
        
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
            # Pick template (F1 or F2 for capitals)
            style_id = random.choice(['F1', 'F2'])
            template = self.templates.get(style_id, 
                'Fill the blank with one word: The capital of {subject} is ____.')
            
            question = template.format(subject=country)
            
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
            style_id = 'F3'
            template = self.templates.get(style_id,
                'Complete with one word: The currency of {subject} is ____.')
            
            question = template.format(subject=country)
            
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
                question = template.format(subject=subject)
                
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
    
    SYSTEM_PROMPT = """You are a creative writing assistant. Your task is to generate short micro-story prompts that have exactly one best one-word completion.

Requirements:
1. Create a 2-sentence micro-story that strongly implies a specific target word
2. End with a blank (____) for the reader to fill
3. The target word should be the single most natural completion
4. Do NOT include the target word anywhere in the prompt
5. Make prompts engaging and varied in theme

Format your response as JSON:
{
  "prompts": [
    {"microstory": "Two sentence story ending with ...", "implied_word": "target"}
  ]
}"""

    def __init__(self, deepseek_client: Optional[DeepSeekClient] = None):
        self.client = deepseek_client or DeepSeekClient()
        self.target_words: List[str] = []
    
    def _get_target_words(self, n: int = 200) -> List[str]:
        """
        Get target words in the Zipf frequency band 3.5-6.0.
        
        Uses wordfreq package.
        """
        try:
            from wordfreq import word_frequency, zipf_frequency
        except ImportError:
            logger.warning("wordfreq not installed, using fallback word list")
            return self._get_fallback_words(n)
        
        # Common English words to check
        from wordfreq import top_n_list
        candidates = top_n_list('en', 5000)
        
        valid_words = []
        min_zipf = CONFIG['wordfreq']['min_zipf']  # 3.5
        max_zipf = CONFIG['wordfreq']['max_zipf']  # 6.0
        
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
    
    def _get_fallback_words(self, n: int) -> List[str]:
        """Fallback word list if wordfreq not available."""
        words = [
            "space", "water", "light", "dark", "time", "life", "death", "love",
            "fear", "hope", "dream", "night", "day", "sun", "moon", "star",
            "fire", "ice", "wind", "rain", "storm", "peace", "war", "truth",
            "lies", "power", "magic", "gold", "silver", "blood", "heart", "soul",
            "mind", "ghost", "shadow", "silence", "music", "dance", "song", "voice",
        ]
        return words[:n]
    
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
        
        for target in self.target_words:
            user_prompt = f"""Generate {candidates_per_target} different micro-story prompts where the answer is "{target}".

Each micro-story should:
- Be 2 sentences long
- End with a blank that "{target}" naturally fills
- NOT contain the word "{target}" anywhere
- Have "{target}" as the single best one-word answer

Return as JSON with "prompts" array."""

            try:
                result = self.client.generate_json(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.7,
                )
                
                prompts = result.get("prompts", [])
                for i, p in enumerate(prompts):
                    microstory = p.get("microstory", "")
                    if microstory and target.lower() not in microstory.lower():
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

Format your response as JSON:
{
  "prompts": [
    {"context": "unusual scenario...", "target_word": "answer"}
  ]
}"""

    def __init__(self, deepseek_client: Optional[DeepSeekClient] = None):
        self.client = deepseek_client or DeepSeekClient()
    
    def generate_candidates(self, n: int = 500) -> List[CandidatePrompt]:
        """
        Generate OOD prompt candidates.
        
        Args:
            n: Number of candidates to generate
            
        Returns:
            List of CandidatePrompt objects
        """
        candidates = []
        batch_size = 10
        n_batches = (n + batch_size - 1) // batch_size
        
        templates = PROMPT_TEMPLATES.get('ood', {})
        
        for batch in range(n_batches):
            user_prompt = f"""Generate {batch_size} out-of-distribution cloze prompts.

Each prompt should:
- Present an unusual, surreal, or pseudo-technical scenario
- End with a blank that has ONE clear best answer
- Feel surprising or unexpected
- NOT be a typical factual or common-sense question

Examples of framing styles:
- Game/puzzle rules
- Ritual/ceremony instructions
- Surreal dreamscape descriptions
- Pseudo-scientific observations
- Mythological or fantastical contexts

Return as JSON with "prompts" array containing objects with "context" and "target_word" fields."""

            try:
                result = self.client.generate_json(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.9,  # Higher for creativity
                )
                
                prompts = result.get("prompts", [])
                for i, p in enumerate(prompts):
                    context = p.get("context", "")
                    target = p.get("target_word", "")
                    
                    if context and target and target.lower() not in context.lower():
                        # Format with OOD template style
                        question = f"Fill the blank with one word: {context} ____."
                        
                        candidates.append(CandidatePrompt(
                            category="ood",
                            question_text=question,
                            target_word=target,
                            target_word_normalized=target.lower(),
                            prompt_style_id=f"ood_{batch}_{i}",
                            source_trace=f"deepseek:ood:batch{batch}",
                            raw_data={"context": context, "target": target},
                        ))
                        
            except Exception as e:
                logger.warning(f"Failed to generate OOD batch {batch}: {e}")
            
            if len(candidates) >= n:
                break
        
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
- Present a well-known English idiom with the last word replaced by a blank
- Have exactly ONE correct answer (the original idiom ending)
- Be recognizable and common idioms

Return JSON: {"prompts": [{"question": "Complete the idiom: bite the ____", "target_word": "bullet"}]}""",

        'facts': """Generate factual one-word answer prompts. Each prompt should:
- Ask about verifiable facts (capitals, currencies, famous people, geography, science)
- Have exactly ONE correct single-word answer
- Be well-known facts that most educated people would know

Return JSON: {"prompts": [{"question": "The capital of Japan is ____.", "target_word": "Tokyo"}]}""",

        'common_sense': """Generate common sense prompts about everyday objects and concepts. Each prompt should:
- Ask about what things are used for, made of, or their properties
- Have exactly ONE natural single-word answer
- Be obvious everyday knowledge

Return JSON: {"prompts": [{"question": "You use scissors to ____.", "target_word": "cut"}]}""",

        'creative': """Generate creative micro-story cloze prompts. Each prompt should:
- Present a 1-2 sentence engaging scenario
- End with a blank that has ONE compelling single-word answer
- The target word should NOT appear in the prompt
- Be imaginative and varied in theme

Return JSON: {"prompts": [{"question": "The astronaut looked back at Earth, feeling an overwhelming sense of ____.", "target_word": "wonder"}]}""",

        'ood': """Generate unusual, out-of-distribution cloze prompts. Each prompt should:
- Present surreal, pseudo-technical, game-like, or ritualistic scenarios
- Have ONE clear best single-word answer despite the unusual framing
- Feel surprising or unexpected compared to typical questions

Return JSON: {"prompts": [{"question": "According to the ancient ritual, before entering the chamber you must whisper ____.", "target_word": "silence"}]}""",
    }
    
    SYSTEM_PROMPT = """You are a prompt generator for a research experiment. Generate high-quality cloze-style prompts with single-word answers.

Requirements:
1. Each prompt must have exactly ONE correct single-word answer
2. The answer must be an English word (alphabetic only)
3. The prompt should NOT contain the answer word
4. Prompts should be diverse and natural

Return your response as valid JSON only."""

    def __init__(self, client: Optional[DeepSeekClient] = None):
        self.client = client or DeepSeekClient()
    
    def generate_for_category(
        self,
        category: str,
        n: int = 100,
        batch_size: int = 20,
    ) -> List[CandidatePrompt]:
        """
        Generate prompts for a specific category.
        
        Args:
            category: Category name (idioms, facts, common_sense, creative, ood)
            n: Number of prompts to generate
            batch_size: Prompts per API call
            
        Returns:
            List of CandidatePrompt objects
        """
        if category not in self.CATEGORY_PROMPTS:
            logger.error(f"Unknown category: {category}")
            return []
        
        category_prompt = self.CATEGORY_PROMPTS[category]
        candidates = []
        n_batches = (n + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            remaining = n - len(candidates)
            if remaining <= 0:
                break
            
            current_batch = min(batch_size, remaining)
            user_prompt = f"Generate {current_batch} prompts.\n\n{category_prompt}"
            
            try:
                result = self.client.generate_json(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    max_tokens=2048,
                )
                
                prompts = result.get("prompts", [])
                for i, p in enumerate(prompts):
                    question = p.get("question", "")
                    target = p.get("target_word", "")
                    
                    # Validate
                    if not question or not target:
                        continue
                    if not target.isalpha():
                        continue
                    if target.lower() in question.lower():
                        continue
                    
                    candidates.append(CandidatePrompt(
                        category=category,
                        question_text=question,
                        target_word=target,
                        target_word_normalized=target.lower(),
                        prompt_style_id=f"deepseek_fallback_{batch_idx}_{i}",
                        source_trace=f"deepseek:fallback:{category}:batch{batch_idx}",
                        raw_data={"question": question, "target": target},
                    ))
                    
            except Exception as e:
                logger.warning(f"Fallback generation failed for {category} batch {batch_idx}: {e}")
        
        logger.info(f"Generated {len(candidates)} fallback prompts for {category}")
        return candidates[:n]


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
        self.creative_gen = CreativeGenerator(self.deepseek_client)
        self.ood_gen = OODGenerator(self.deepseek_client)
        self.fallback_gen = DeepSeekFallbackGenerator(self.deepseek_client)
    
    def generate_all(
        self,
        prompts_per_category: int = 500,
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
        
        # Category A: Idioms
        logger.info("Generating idiom candidates...")
        results['idioms'] = self.idiom_gen.generate_candidates(prompts_per_category)
        
        # Category B: Facts
        logger.info("Generating fact candidates...")
        results['facts'] = self.fact_gen.generate_candidates(prompts_per_category)
        
        # Category C: Common Sense
        logger.info("Generating common sense candidates...")
        results['common_sense'] = self.common_sense_gen.generate_candidates(prompts_per_category)
        
        if not skip_generated:
            # Category D: Creative
            logger.info("Generating creative candidates...")
            results['creative'] = self.creative_gen.generate_candidates(prompts_per_category)
            
            # Category E: OOD
            logger.info("Generating OOD candidates...")
            results['ood'] = self.ood_gen.generate_candidates(prompts_per_category)
        
        # Fill gaps with fallback if enabled
        if use_fallback:
            for category, candidates in results.items():
                gap = prompts_per_category - len(candidates)
                if gap > 0:
                    logger.info(f"Category {category} has {len(candidates)} prompts, need {gap} more. Using DeepSeek fallback...")
                    fallback_prompts = self.fallback_gen.generate_for_category(category, gap)
                    results[category].extend(fallback_prompts)
                    logger.info(f"Category {category} now has {len(results[category])} prompts")
        
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
    
    print("\n" + "=" * 60)
    print("Data mining tests complete!")
    print("=" * 60)
