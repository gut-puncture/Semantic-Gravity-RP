"""
Semantic Gravity Experiment - Source Package

This package contains modules for running the mechanistic negative-instruction
study on Qwen models.

Modules:
- config: Global configuration, paths, and prompt templates
- utils: Shared utilities (normalization, seeding, model loading)
- api_clients: External API clients (DeepSeek, Wikidata, ConceptNet)
- data_mining: Category-specific prompt generators
- validator: Prompt validation and selection
"""

from .config import CONFIG, PROMPT_TEMPLATES, setup_directories, get_base_paths
from .utils import (
    normalize_for_match,
    word_in_text,
    find_word_occurrences,
    generate_surface_variants,
    set_seed,
    set_all_seeds,
    ModelWrapper,
    setup_logging,
    compute_token_char_spans,
    map_word_to_tokens,
)
from .api_clients import (
    DeepSeekClient,
    WikidataClient,
    ConceptNetClient,
    download_idioms_csv,
)
from .data_mining import (
    CandidatePrompt,
    IdiomGenerator,
    FactGenerator,
    CommonSenseGenerator,
    CreativeGenerator,
    OODGenerator,
    DeepSeekFallbackGenerator,
    DatasetGenerator,
)
from .validator import (
    ValidationResult,
    ValidatedPrompt,
    PromptValidator,
    PromptSelector,
    TargetTracker,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    'CONFIG',
    'PROMPT_TEMPLATES',
    'setup_directories',
    'get_base_paths',
    # Utils
    'normalize_for_match',
    'word_in_text',
    'find_word_occurrences',
    'generate_surface_variants',
    'set_seed',
    'set_all_seeds',
    'ModelWrapper',
    'setup_logging',
    'compute_token_char_spans',
    'map_word_to_tokens',
    # API Clients
    'DeepSeekClient',
    'WikidataClient',
    'ConceptNetClient',
    'download_idioms_csv',
    # Data Mining
    'CandidatePrompt',
    'IdiomGenerator',
    'FactGenerator',
    'CommonSenseGenerator',
    'CreativeGenerator',
    'OODGenerator',
    'DatasetGenerator',
    # Validator
    'ValidationResult',
    'ValidatedPrompt',
    'PromptValidator',
    'PromptSelector',
    'TargetTracker',
]
