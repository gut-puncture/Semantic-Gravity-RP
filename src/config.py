"""
config.py - Global Configuration for Semantic Gravity Experiment

This module defines:
- Paths for Google Drive and local output directories
- Model configuration
- Seeds for reproducibility
- Prompt templates

Designed to work both locally (for dataset construction) and in Colab (for inference).
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

def get_base_paths() -> dict:
    """
    Get base paths depending on environment.
    
    Returns:
        dict with 'drive_root', 'data_root', 'output_root', 'model_path'
    """
    if is_colab():
        drive_root = Path("/content/drive/MyDrive/SemanticGravity")
        return {
            'drive_root': drive_root,
            'data_root': drive_root / "data",
            'output_root': drive_root / "outputs",
            'model_path': Path("/content/drive/MyDrive/models/Qwen2.5-7B-Instruct"),
            'assets_root': drive_root / "assets",
        }
    else:
        # Local development - use project directory
        project_root = Path(__file__).parent.parent
        return {
            'drive_root': project_root,  # For local testing
            'data_root': project_root / "data",
            'output_root': project_root / "outputs",
            'model_path': None,  # Model not needed locally
            'assets_root': project_root / "assets",
        }

# ============================================================================
# GLOBAL CONFIG DICTIONARY
# ============================================================================

CONFIG = {
    # ----- Reproducibility -----
    'seeds': {
        'python': 42,
        'numpy': 42,
        'torch': 42,
    },
    
    # ----- Model Configuration -----
    'model': {
        'model_id': 'Qwen/Qwen2.5-7B-Instruct',
        'torch_dtype': 'bfloat16',
        'device_map': 'auto',
        'add_special_tokens': False,
        'max_new_tokens_greedy': 8,
        'max_new_tokens_stochastic': 10,
        'num_stochastic_samples': 16,
        'temperature': 1.0,
        'top_p': 0.9,
    },
    
    # ----- Dataset Configuration -----
    'dataset': {
        'total_prompts': 2500,
        'num_categories': 5,
        'prompts_per_category': 500,
        'categories': ['idioms', 'facts', 'common_sense', 'creative', 'ood'],
        'candidate_multiplier': 2,
        'pressure_bins': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'min_pressure_threshold': 0.20,
        'candidates_per_target': 3,
        'creative_ood_target_count': 600,
        'max_target_repetition': 2,  # Same word in at most 2 categories
        # GPT-5.2 fallback tuning (when sources are down or sparse)
        'fallback_max_extra_batches': 3,
        'common_sense_fallback_max_target_repeats': 1,
        'common_sense_fallback_max_subject_repeats': 1,
    },
    
    # ----- Validation Configuration -----
    'validation': {
        'min_naturalness_score': 7,
        'max_ambiguity_score': 3,
    },
    
    # ----- OpenAI GPT-5.2 API Configuration -----
    'openai': {
        'base_url': 'https://api.openai.com/v1',
        'retry_attempts': 3,
        'retry_base_delay': 1.0,
        'retry_max_delay': 30.0,
        'json_retry_attempts': 4,
        'model': 'gpt-5.2-2025-12-11',
        'max_output_tokens_generation': 2000,
        'max_output_tokens_validation': 2000,
        'text_format': {"type": "json_object"},
        'reasoning': {"effort": "none", "summary": "auto"},
        'verbosity': "low",
        'store': True,
        'batch_completion_window': "24h",
        'request_timeout_seconds': 180,
    },
    
    # ----- Data Sources -----
    'sources': {
        'idioms_repo': 'baiango/english_idioms',
        'idioms_file': 'idioms.csv',
        'wikidata_endpoint': 'https://query.wikidata.org/sparql',
        'conceptnet_endpoint': 'https://api.conceptnet.io',
        'writingprompts_dataset': 'euclaise/writingprompts',
    },
    
    # ----- Word Frequency -----
    'wordfreq': {
        'min_zipf': 3.5,
        'max_zipf': 6.0,
    },
    
    # ----- Error Tolerance -----
    'error_tolerance': {
        'max_mapping_error_rate': 0.001,  # 0.1%
        'max_mapping_error_count': 0,  # Stricter: zero tolerance
    },
    
    # ----- Bootstrap Configuration -----
    'bootstrap': {
        'n_iterations': 1000,
        'ci_percentiles': [2.5, 97.5],
    },
    
    # ----- Patching Configuration -----
    'patching': {
        'prompts_per_bin': 50,
        'total_subset_size': 250,
        'min_failures': 30,
    },
}

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PROMPT_TEMPLATES = {
    'baseline': (
        "Answer with exactly one English word.\n"
        "Question: {question}\n"
        "Answer:"
    ),
    
    'negative_instruction': (
        "Answer with exactly one English word.\n"
        "Do not use the word \"{target}\" anywhere in your answer.\n"
        "Question: {question}\n"
        "Answer:"
    ),
    
    # ----- Category-specific templates -----
    'idioms': {
        'I1': 'Complete the idiom with one word: "{idiom_with_blank}"',
        'I2': 'Fill the blank (one word): {idiom_with_blank}',
        'I3': 'Idiom completion (one word): {idiom_with_blank}',
    },
    
    'facts': {
        'F1': 'Fill the blank with one word: The capital of {subject} is ____.',
        'F2': "One-word answer: {subject}'s capital is ____.",
        'F3': 'Complete with one word: The currency of {subject} is ____.',
    },
    
    'common_sense': {
        'C1_UsedFor': 'Fill the blank (one word): You use a {subject} to ____.',
        'C2_MadeOf': 'Fill the blank (one word): A {subject} is made of ____.',
        'C3_HasProperty': 'Fill the blank (one word): A typical {subject} is ____.',
    },
    
    'creative': {
        'default': 'Fill the blank with one word: {microstory} ____.',
    },
    
    'ood': {
        'O1': 'Fill the blank with one word: In this game, the only winning move is ____.',
        'O2': 'Fill the blank with one word: According to the ritual manual, you must say ____ before you enter.',
    },
}

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def setup_directories(run_timestamp: Optional[str] = None) -> dict:
    """
    Create all necessary directories for the experiment.
    
    Args:
        run_timestamp: Optional timestamp for run-specific output folder.
                      If None, uses current datetime.
    
    Returns:
        dict with all created directory paths
    """
    paths = get_base_paths()
    
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Define directory structure
    directories = {
        # Data directories
        'data_raw': paths['data_root'] / 'raw',
        'data_candidates': paths['data_root'] / 'candidates',
        'data_validated': paths['data_root'] / 'validated',
        
        # Run-specific output directory
        'run_root': paths['output_root'] / f'experiment_run_{run_timestamp}',
    }
    
    # Add run subdirectories
    run_root = directories['run_root']
    directories.update({
        'runs_data': run_root / 'data',
        'runs_traces': run_root / 'runs',
        'runs_mechanistic': run_root / 'runs' / 'mechanistic_trace',
        'runs_samples': run_root / 'runs' / 'failure_samples',
        'figures': run_root / 'figures',
        'appendix': run_root / 'appendix_examples',
        'errors': run_root / 'errors',
    })
    
    # Also add assets directory
    directories['assets'] = paths['assets_root']
    
    # Create all directories
    for dir_name, dir_path in directories.items():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories

# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

def validate_environment() -> dict:
    """
    Validate the environment and log important metadata.
    
    Returns:
        dict with environment metadata
    """
    import sys
    from datetime import datetime
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'is_colab': is_colab(),
    }
    
    # Check for key packages
    packages_to_check = [
        'torch', 'transformers', 'accelerate', 'numpy', 
        'pandas', 'scipy', 'matplotlib', 'tqdm', 'requests'
    ]
    
    package_versions = {}
    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            package_versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            package_versions[pkg] = 'not installed'
    
    metadata['package_versions'] = package_versions
    
    # GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            metadata['gpu_available'] = True
            metadata['gpu_name'] = torch.cuda.get_device_name(0)
            metadata['gpu_count'] = torch.cuda.device_count()
        else:
            metadata['gpu_available'] = False
    except ImportError:
        metadata['gpu_available'] = False
    
    # Check transformers version
    try:
        import transformers
        version = transformers.__version__
        major, minor = map(int, version.split('.')[:2])
        metadata['transformers_compatible'] = (major >= 4 and minor >= 37)
        if not metadata['transformers_compatible']:
            print(f"⚠️ Warning: transformers version {version} < 4.37.0 may not support all features")
    except Exception as e:
        metadata['transformers_compatible'] = False
        print(f"⚠️ Warning: Could not check transformers version: {e}")
    
    return metadata

# ============================================================================
# DRIVE MOUNTING (COLAB ONLY)
# ============================================================================

def mount_drive():
    """
    Mount Google Drive in Colab. No-op if not in Colab.
    """
    if is_colab():
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
    else:
        print("ℹ️ Not in Colab, skipping Drive mount")

# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================

def print_config():
    """Pretty print the current configuration."""
    import json
    print("=" * 60)
    print("SEMANTIC GRAVITY EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(json.dumps(CONFIG, indent=2, default=str))
    print("=" * 60)


if __name__ == "__main__":
    # Quick validation when run directly
    print("Running config validation...")
    metadata = validate_environment()
    print(f"\nEnvironment: {'Colab' if metadata['is_colab'] else 'Local'}")
    print(f"Python: {metadata['python_version'].split()[0]}")
    print(f"GPU: {metadata.get('gpu_name', 'N/A')}")
    print("\nPackage versions:")
    for pkg, ver in metadata['package_versions'].items():
        print(f"  {pkg}: {ver}")
