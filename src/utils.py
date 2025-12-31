"""
utils.py - Shared Utilities for Semantic Gravity Experiment

This module provides:
- Text normalization for word matching
- Deterministic seeding for reproducibility  
- Model loading with singleton pattern (for Colab inference)
- Logging configuration
- Common helper functions

Designed to work both locally (for dataset construction) and in Colab (for inference).
"""

import os
import re
import string
import unicodedata
import logging
from typing import Optional, List, Tuple, Set
from pathlib import Path

# Configure module-level logger
logger = logging.getLogger(__name__)

# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

# Build punctuation translation table once
# Include: . , ! ? ; : " ' ( ) [ ] { } < > / \ | - – — _
_PUNCTUATION_CHARS = ".,!?;:\"'()[]{}<>/\\|-–—_"
_PUNCT_TO_SPACE = str.maketrans(_PUNCTUATION_CHARS, ' ' * len(_PUNCTUATION_CHARS))


def normalize_for_match(text: str) -> str:
    """
    Normalize text for word-level matching.
    
    Steps (per specification Section 5.2):
    1. Unicode NFKC normalization
    2. Lowercase
    3. Replace punctuation with spaces
    4. Collapse multiple whitespace to single spaces
    5. Strip leading/trailing whitespace
    
    Args:
        text: Input text string
        
    Returns:
        Normalized string for comparison
    """
    # Step 1: Unicode NFKC
    text = unicodedata.normalize('NFKC', text)
    
    # Step 2: Lowercase
    text = text.lower()
    
    # Step 3: Replace punctuation with spaces
    text = text.translate(_PUNCT_TO_SPACE)
    
    # Step 4 & 5: Collapse whitespace and strip
    text = ' '.join(text.split())
    
    return text


def word_in_text(target: str, text: str) -> bool:
    """
    Check if target word appears in text as a complete word.
    
    Uses normalization, then checks for exact word match.
    
    Args:
        target: The target word to find
        text: The text to search in
        
    Returns:
        True if target appears as a complete word in text
    """
    target_norm = normalize_for_match(target)
    text_norm = normalize_for_match(text)
    text_words = text_norm.split()
    return target_norm in text_words


def find_word_occurrences(target: str, text: str) -> List[Tuple[int, int]]:
    """
    Find all character-level occurrences of target word in text.

    Matches are case-insensitive and require non-alphanumeric boundaries
    (as defined by str.isalnum()).
    This avoids false positives like "space2" while still matching "space."

    Args:
        target: The target word to find
        text: The text to search in

    Returns:
        List of (start, end) character indices for each occurrence
    """
    if not target:
        return []

    occurrences = []
    pattern = re.compile(re.escape(target), flags=re.IGNORECASE)
    for match in pattern.finditer(text):
        start, end = match.span()
        left_ok = start == 0 or not text[start - 1].isalnum()
        right_ok = end >= len(text) or not text[end].isalnum()
        if left_ok and right_ok:
            occurrences.append((start, end))

    return occurrences


# ============================================================================
# SURFACE VARIANT GENERATION (for tokenization enumeration)
# ============================================================================

def generate_surface_variants(word: str) -> Set[str]:
    """
    Generate surface variants of a word for tokenization enumeration.
    
    Per specification Section 6.1, generates:
    - Case variants: lower, title, upper, original
    - Whitespace variants: with/without leading/trailing space
    - Punctuation variants: with/without trailing period
    
    Args:
        word: The base word
        
    Returns:
        Set of all surface variant strings
    """
    variants = set()
    
    # Case variants
    case_variants = [
        word,
        word.lower(),
        word.title(),
        word.upper(),
    ]
    
    # Remove duplicates while preserving all unique variants
    case_variants = list(set(case_variants))
    
    for v in case_variants:
        # Whitespace variants
        whitespace_forms = [
            v,           # no extra whitespace
            f" {v}",     # leading space
            f"{v} ",     # trailing space
            f" {v} ",    # both
        ]
        
        for form in whitespace_forms:
            variants.add(form)
            # Trailing period variants
            variants.add(form.rstrip() + ".")
    
    return variants


# ============================================================================
# DETERMINISTIC SEEDING
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: The seed value to use
    """
    import random
    try:
        import numpy as np
    except ImportError:
        np = None
        logger.warning("NumPy not available, skipping numpy seed setting")

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # For extra reproducibility (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.warning("PyTorch not available, skipping torch seed setting")


def set_all_seeds() -> None:
    """
    Set all seeds from CONFIG.
    
    Uses the seed values defined in config.py.
    """
    from .config import CONFIG
    
    seeds = CONFIG['seeds']
    
    import random
    try:
        import numpy as np
    except ImportError:
        np = None
        logger.warning("NumPy not available, skipping numpy seed setting")

    random.seed(seeds['python'])
    if np is not None:
        np.random.seed(seeds['numpy'])
    
    try:
        import torch
        torch.manual_seed(seeds['torch'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds['torch'])
    except ImportError:
        logger.warning("PyTorch not available, skipping torch seed setting")


# ============================================================================
# MODEL WRAPPER (SINGLETON PATTERN)
# ============================================================================

class ModelWrapper:
    """
    Singleton wrapper for Qwen model and tokenizer.
    
    Loads the model only once and reuses it across calls.
    Designed for use in Colab with GPU.
    
    Usage:
        wrapper = ModelWrapper.get_instance()
        outputs = wrapper.generate(prompt, max_new_tokens=8)
    """
    
    _instance: Optional['ModelWrapper'] = None
    
    def __init__(self):
        """Initialize the model wrapper. Use get_instance() instead."""
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    @classmethod
    def get_instance(cls) -> 'ModelWrapper':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing or reloading)."""
        if cls._instance is not None:
            cls._instance.unload()
            cls._instance = None
    
    def load(self, model_path: Optional[str] = None, force_reload: bool = False) -> None:
        """
        Load the model and tokenizer.
        
        Args:
            model_path: Path to model. If None, uses CONFIG path.
            force_reload: If True, reload even if already loaded.
        """
        if self._loaded and not force_reload:
            logger.info("Model already loaded, skipping")
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Get model path from env/config if not provided
        if model_path is None:
            env_path = os.environ.get("SEMANTIC_GRAVITY_MODEL_PATH") or os.environ.get("MODEL_PATH")
            if env_path:
                model_path = env_path
            else:
                from .config import get_base_paths, CONFIG
                paths = get_base_paths()
                model_path = paths.get('model_path')
                if model_path is None:
                    model_path = CONFIG['model']['model_id']
        
        logger.info(f"Loading model from: {model_path}")
        
        # Check transformers version
        import transformers
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])
        if major < 4 or (major == 4 and minor < 37):
            logger.warning(f"transformers {version} may not support all features. Recommend >= 4.37.0")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Resolve device and dtype
        env_device = os.environ.get("SEMANTIC_GRAVITY_DEVICE") or os.environ.get("MODEL_DEVICE")
        config_device = None
        try:
            from .config import CONFIG
            config_device = CONFIG.get("model", {}).get("device")
        except Exception:
            config_device = None

        device = env_device or config_device or "auto"
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype_env = os.environ.get("SEMANTIC_GRAVITY_TORCH_DTYPE") or os.environ.get("MODEL_TORCH_DTYPE")
        dtype_cfg = None
        try:
            from .config import CONFIG
            dtype_cfg = CONFIG.get("model", {}).get("torch_dtype")
        except Exception:
            dtype_cfg = None

        dtype_raw = dtype_env or dtype_cfg or "bfloat16"
        if isinstance(dtype_raw, str):
            dtype_key = dtype_raw.strip().lower()
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            torch_dtype = dtype_map.get(dtype_key, torch.bfloat16)
        else:
            torch_dtype = dtype_raw

        if device == "mps" and torch_dtype == torch.bfloat16:
            logger.warning("bfloat16 not supported on MPS; switching to float16")
            torch_dtype = torch.float16

        device_map = None
        try:
            from .config import CONFIG
            device_map = CONFIG.get("model", {}).get("device_map", "auto")
        except Exception:
            device_map = "auto"

        env_device_map = os.environ.get("SEMANTIC_GRAVITY_DEVICE_MAP")
        if env_device_map:
            device_map = env_device_map

        # Load model
        logger.info("Loading model (this may take a few minutes)...")
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        if device.startswith("cuda"):
            model_kwargs["device_map"] = device_map or "auto"
        else:
            model_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if not device.startswith("cuda") and device != "cpu":
            self.model.to(device)

        # Ensure eager attention when we need output_attentions from generate().
        # Some backends (sdpa) do not return attentions.
        try:
            if hasattr(self.model, "set_attn_implementation"):
                self.model.set_attn_implementation("eager")
            if hasattr(self.model, "config"):
                self.model.config.attn_implementation = "eager"
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.attn_implementation = "eager"
        except Exception as e:
            logger.warning("Failed to set eager attention implementation: %s", e)

        self.model.eval()  # Set to evaluation mode
        self._loaded = True
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Vocab size: {len(self.tokenizer)}")
        logger.info(f"   Device: {next(self.model.parameters()).device}")
    
    def unload(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._loaded = False
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Model unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def tokenize(self, text: str, **kwargs) -> dict:
        """
        Tokenize input text.
        
        Args:
            text: Input text
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Tokenizer output dict
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if "add_special_tokens" not in kwargs:
            try:
                from .config import CONFIG
                kwargs["add_special_tokens"] = CONFIG.get("model", {}).get("add_special_tokens", False)
            except Exception:
                kwargs["add_special_tokens"] = False

        return self.tokenizer(
            text,
            return_tensors="pt",
            **kwargs
        ).to(self.model.device)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 8,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict_in_generate: bool = False,
    ) -> dict:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling (False=greedy)
            num_return_sequences: Number of sequences to return
            temperature: Sampling temperature (only if do_sample=True)
            top_p: Nucleus sampling parameter (only if do_sample=True)
            output_hidden_states: Return hidden states
            output_attentions: Return attention weights
            return_dict_in_generate: Return GenerateOutput object
            
        Returns:
            Dict with 'generated_ids', 'generated_text', and optionally
            'hidden_states', 'attentions'
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        import torch
        
        # Tokenize
        inputs = self.tokenize(prompt)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Extract generated portion
        generated_ids = outputs.sequences[:, input_length:]
        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        result = {
            'generated_ids': generated_ids,
            'generated_text': generated_texts if num_return_sequences > 1 else generated_texts[0],
            'full_ids': outputs.sequences,
        }
        
        if output_hidden_states and hasattr(outputs, 'hidden_states'):
            result['hidden_states'] = outputs.hidden_states
        
        if output_attentions and hasattr(outputs, 'attentions'):
            result['attentions'] = outputs.attentions
        
        return result
    
    def get_logits(self, input_ids) -> 'torch.Tensor':
        """
        Get logits for input tokens.
        
        Args:
            input_ids: Token IDs tensor
            
        Returns:
            Logits tensor
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        import torch
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        return outputs.logits


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_str: Optional[str] = None,
) -> None:
    """
    Configure logging for the experiment.
    
    Args:
        level: Logging level
        log_file: Optional file to write logs to
        format_str: Optional custom format string
    """
    if format_str is None:
        format_str = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True,
    )


# ============================================================================
# PREFIX INCREMENTAL DECODE FOR TOKEN MAPPING
# ============================================================================

def compute_token_char_spans(
    token_ids: List[int],
    tokenizer,
) -> List[Tuple[int, int]]:
    """
    Compute character spans for each token using prefix incremental decoding.
    
    Per specification Section 5.3:
    - Decode ids[0:i] for each i
    - Token i spans (len(decode[0:i-1]), len(decode[0:i]))
    
    Args:
        token_ids: List of token IDs
        tokenizer: The tokenizer instance
        
    Returns:
        List of (start, end) character indices for each token
    """
    spans = []
    prev_length = 0
    
    for i in range(len(token_ids)):
        decoded = tokenizer.decode(
            token_ids[:i+1],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )
        curr_length = len(decoded)
        spans.append((prev_length, curr_length))
        prev_length = curr_length
    
    return spans


def map_word_to_tokens(
    target: str,
    token_ids: List[int],
    tokenizer,
    full_text: str,
) -> List[Tuple[int, int, int, int]]:
    """
    Map word occurrences in text to token indices.
    
    Args:
        target: Target word to find
        token_ids: Generated token IDs
        tokenizer: Tokenizer instance
        full_text: Full decoded text (for verification)
        
    Returns:
        List of (token_start, token_end, char_start, char_end) for each occurrence
        Returns empty list if word not found
    """
    # Find occurrences in text
    occurrences = find_word_occurrences(target, full_text)
    
    if not occurrences:
        return []
    
    # Get token spans
    token_spans = compute_token_char_spans(token_ids, tokenizer)
    
    mappings = []
    
    for char_start, char_end in occurrences:
        # Find tokens that cover this character range
        token_start = None
        token_end = None
        
        for tok_idx, (tok_start, tok_end) in enumerate(token_spans):
            # Token overlaps with match if ranges intersect
            if tok_end > char_start and tok_start < char_end:
                if token_start is None:
                    token_start = tok_idx
                token_end = tok_idx
        
        if token_start is not None:
            mappings.append((token_start, token_end, char_start, char_end))
    
    return mappings


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def ensure_list(x):
    """Ensure input is a list."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if hasattr(x, '__iter__'):
        return list(x)
    return [x]


def truncate_string(s: str, max_len: int = 50) -> str:
    """Truncate string with ellipsis if too long."""
    if len(s) <= max_len:
        return s
    return s[:max_len-3] + "..."


# ============================================================================
# RUN ROOT RESOLUTION (centralized for all modules)
# ============================================================================

def resolve_run_root(output_root: Optional[Path]) -> Path:
    """
    Resolve the run root directory from output_root or find the latest run.

    This function is centralized here to avoid duplication across modules
    (behavior_analysis.py, metrics_attn.py, patching.py).

    Priority order:
    1) If output_root name starts with experiment_run_, return it (specific run).
    2) If output_root contains experiment_run_* subdirs, select latest by name.
    3) If output_root has runs/ subdir (but no experiment_run_* children), use it.
    4) Otherwise, return output_root as-is (backward compatibility).
    5) If output_root is None, select latest experiment_run_* under base outputs.

    Args:
        output_root: Optional run root or base outputs directory

    Returns:
        Resolved run root Path
    """
    try:
        from .config import get_base_paths
    except ImportError:
        from config import get_base_paths

    paths = get_base_paths()
    base_out = paths.get("output_root", Path("outputs"))

    def find_latest_run_in(parent: Path) -> Optional[Path]:
        """Find latest experiment_run_* subdir by lexicographic name."""
        if not parent.exists() or not parent.is_dir():
            return None
        try:
            run_dirs = sorted(
                [d for d in parent.iterdir() if d.is_dir() and d.name.startswith("experiment_run_")],
                key=lambda d: d.name,
            )
            return run_dirs[-1] if run_dirs else None
        except OSError:
            return None

    if output_root is not None:
        candidate = Path(output_root)

        # Priority 1: If name starts with experiment_run_, it IS a run root
        if candidate.name.startswith("experiment_run_"):
            return candidate

        # Priority 2: Check for experiment_run_* subdirs BEFORE checking runs/
        latest = find_latest_run_in(candidate)
        if latest:
            logger.warning(
                "Selecting latest experiment_run_* under %s: %s",
                candidate, latest,
            )
            return latest

        # Priority 3: If it has runs/ but no experiment_run_*, treat as run root
        if candidate.exists() and (candidate / "runs").exists():
            return candidate

        # Priority 4: Backward compatibility - return as-is
        return candidate

    # output_root is None: find latest run under base_out
    latest = find_latest_run_in(base_out)
    if latest:
        logger.warning("output_root not provided; using latest run dir: %s", latest)
        return latest

    # Fallback to base_out
    return base_out


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    # Run basic unit tests
    print("=" * 60)
    print("Running utils.py unit tests...")
    print("=" * 60)
    
    # Test normalization
    print("\n1. Testing normalize_for_match:")
    test_cases = [
        ("Space", "space"),
        ("SPACE", "space"),
        ("space.", "space"),
        ("space!", "space"),
        ("  space  ", "space"),
        ("hello world", "hello world"),
        ("hello,world", "hello world"),
        ("hello—world", "hello world"),  # em dash
    ]
    for input_str, expected in test_cases:
        result = normalize_for_match(input_str)
        status = "✅" if result == expected else "❌"
        print(f"   {status} normalize_for_match('{input_str}') = '{result}' (expected: '{expected}')")
    
    # Test word_in_text
    print("\n2. Testing word_in_text:")
    test_cases = [
        ("space", "I love space!", True),
        ("space", "spacetime is cool", False),
        ("space", "Space is vast", True),
        ("space", "aerospace", False),
        ("space", "space.", True),  # With punctuation
    ]
    for target, text, expected in test_cases:
        result = word_in_text(target, text)
        status = "✅" if result == expected else "❌"
        print(f"   {status} word_in_text('{target}', '{text}') = {result} (expected: {expected})")
    
    # Test surface variants
    print("\n3. Testing generate_surface_variants:")
    variants = generate_surface_variants("space")
    print(f"   Generated {len(variants)} variants for 'space':")
    for v in sorted(variants)[:10]:
        print(f"      '{v}'")
    if len(variants) > 10:
        print(f"      ... and {len(variants) - 10} more")
    
    # Test find_word_occurrences
    print("\n4. Testing find_word_occurrences:")
    test_cases = [
        ("space", "I love space and space travel", [(7, 12), (17, 22)]),
        ("space", "Space is vast", [(0, 5)]),
        ("space", "spacetime", []),
        ("space", "space2", []),
        ("space", "space-time", [(0, 5)]),
    ]
    for target, text, expected in test_cases:
        result = find_word_occurrences(target, text)
        status = "✅" if result == expected else "❌"
        print(f"   {status} find_word_occurrences('{target}', '{text}') = {result}")
    
    # Test seeding
    print("\n5. Testing set_seed:")
    import random
    set_seed(42)
    r1 = random.random()
    set_seed(42)
    r2 = random.random()
    status = "✅" if r1 == r2 else "❌"
    print(f"   {status} Seeds produce reproducible results: {r1} == {r2}")
    
    print("\n" + "=" * 60)
    print("Unit tests complete!")
    print("=" * 60)
