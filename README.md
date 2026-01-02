# Semantic Gravity Wells: Why Negative Constraints Backfire

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Abstract

This repository contains the complete codebase, data, and analysis pipeline for reproducing the research paper "The Forbidden Word Effect." The paper presents the first comprehensive mechanistic investigation of why large language models fail to follow negative instructions (e.g., "Do not use the word Paris in your answer").

**Key Findings:**
- Violation probability follows a tight logistic relationship with semantic pressure (baseline target probability)
- Negative instructions induce 4.4× weaker suppression in failures vs. successes
- 87.5% of failures exhibit a "priming signature" where the model attends more to the forbidden word than the negation cue
- Late-layer FFN contributions in failures are 4× larger than in successes
- Layers 23–27 are causally responsible for override failures (confirmed via activation patching)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Structure](#repository-structure)
3. [Methodology Overview](#methodology-overview)
4. [Reproducing the Experiments](#reproducing-the-experiments)
5. [Data Pipeline](#data-pipeline)
6. [Analysis Components](#analysis-components)
7. [Hardware Requirements](#hardware-requirements)
8. [Citation](#citation)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/gut-puncture/Semantic-Gravity-RP.git
cd Semantic-Gravity-RP

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OpenAI for prompt generation)

# 5. Run the analysis pipeline (requires GPU)
# See "Reproducing the Experiments" section below
```

---

## Repository Structure

```
Semantic-Gravity-RP/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
│
├── research_paper/              # Paper materials
│   ├── latex/                   # LaTeX source files
│   │   ├── paper_draft.tex      # Main paper source
│   │   └── neurips_2024.sty     # NeurIPS style file
│   ├── figures/                 # Publication-ready figures (PDF + PNG)
│   ├── output/                  # Compiled PDFs and aux files
│   └── scripts/                 # Figure generation scripts
│
├── src/                         # Core source code
│   ├── config.py                # Configuration and constants
│   ├── utils.py                 # Utilities and ModelWrapper
│   ├── prompt_builder.py        # Prompt construction
│   ├── detector.py              # Violation detection
│   ├── validator.py             # Prompt validation with GPT-5.2
│   ├── data_mining.py           # Prompt generation pipelines
│   ├── dataset_pipeline.py      # Dataset finalization with P_sem
│   ├── runner.py                # Experiment runner
│   ├── metrics_psem.py          # Semantic pressure computation
│   ├── metrics_attn.py          # Attention metrics, logit lens, decomposition
│   └── patching.py              # Activation patching
│
├── notebooks/                   # Jupyter notebooks
│   └── semantic_gravity.ipynb   # Full GPU pipeline (for A100 Colab/VM)
│
├── data/                        # Data files
│   ├── raw/                     # Raw source data
│   ├── candidates/              # Generated prompt candidates
│   ├── validated/               # GPT-5.2 validated prompts
│   ├── validated_single_token/  # Single-token target prompts
│   └── prompts.csv              # Final 2,500 experiment prompts
│
├── outputs/                     # Experiment outputs
│   └── experiment_run_YYYYMMDD_HHMM/
│       ├── runs/                # Raw experiment outputs
│       │   ├── completions_greedy.jsonl
│       │   ├── completions_samples.jsonl
│       │   ├── detection_mapping.jsonl
│       │   ├── attention_metrics.csv
│       │   ├── logit_lens.csv
│       │   ├── ffn_attn_decomp.csv
│       │   └── patching_results.csv
│       ├── figures/             # Generated analysis figures
│       └── logs/                # Run logs
│
└── docs/                        # Planning and specification documents
    ├── specification.md         # Full experimental specification
    ├── implementation-plan.md   # Implementation details
    ├── execution-plan.md        # Execution timeline
    └── GPU_PIPELINE_IMPLEMENTATION_PLAN.md  # GPU optimization notes
```

---

## Methodology Overview

### 1. Dataset Construction

We construct a dataset of **2,500 prompts** (500 per category) designed to elicit single-word completions:

| Category | Description | Example |
|----------|-------------|---------|
| **Idioms** | Partial idioms with unique completions | "Spill the ____" → *beans* |
| **Factual** | World knowledge with definitive answers | "The capital of France is ____" → *Paris* |
| **Common Sense** | Everyday knowledge with strong expectations | "A hammer is used to drive in ____" → *nails* |
| **Creative** | Context-driven completions | "The color of the sky at noon is ____" → *blue* |
| **Out-of-Distribution** | Varied pressure levels with surreal contexts | "The flame that consumes ____ of its shadow is crowned Protocol-Sovereign" → *most* |

### 2. Semantic Pressure (P₀)

For each prompt, we compute the **baseline probability** of the target word without any negative constraint:

```
P₀ = Σ Π P(sᵢ | context, s<ᵢ) for all valid tokenizations of target X
```

This captures how strongly the model "wants" to produce the forbidden word.

### 3. Experimental Conditions

Each prompt is evaluated under two conditions:
- **Baseline**: "Answer with one word. Question: [prompt]"
- **Negative Instruction**: "Answer with one word. Do not use the word '[X]' in your answer. Question: [prompt]"

### 4. Behavioral Analysis

- **16 stochastic samples** per prompt (temperature=1.0, top-p=0.9)
- **Violation detection** using tokenizer-aware word boundary matching
- **Logistic regression** of violation rate against P₀

### 5. Mechanistic Analysis

For each prompt:
- **Attention metrics**: Instruction Attention Ratio, Negation Focus, Target-Mention Focus, Priming Index
- **Logit lens**: Layer-wise probability trajectories
- **FFN/Attention decomposition**: Component contributions to target probability
- **Activation patching**: Causal interventions on layers 0-27

---

## Reproducing the Experiments

### Stage 1: Prompt Generation and Validation (CPU)

```bash
# Generate prompt candidates using GPT-5.2
python -m src.data_mining --stage generate

# Validate prompts with GPT-5.2 scoring
python -m src.data_mining --stage validate

# This produces data/validated/*.jsonl
```

### Stage 2: GPU Pipeline (A100 recommended)

The full pipeline runs in `notebooks/semantic_gravity.ipynb` on Google Colab or a GPU VM.
For Colab-specific setup, see `docs/colab.md`.

**Steps:**
1. Mount Google Drive with model files and validated prompts
2. Filter to single-token targets
3. Compute P_sem (semantic pressure) for all prompts
4. Run mechanistic passes (greedy generation with hidden states)
5. Run behavioral passes (16 samples per prompt)
6. Detection and mapping
7. Compute metrics (attention, logit lens, decomposition)
8. Activation patching
9. Bootstrap confidence intervals
10. Generate figures

### Stage 3: Figure Generation and Analysis

```bash
# Generate publication-ready figures
cd research_paper/scripts
python generate_figures.py --input ../../outputs/experiment_run_YYYYMMDD_HHMM/
```

---

## Data Pipeline

### Prompt Validation Criteria

Each prompt is scored by GPT-5.2 on four criteria:
1. **Unique best answer**: Target X is the clear dominant response
2. **Low ambiguity**: Score 1-10 (lower is better)
3. **No answer leakage**: Prompt does not contain X
4. **Naturalness**: Score 1-10 (higher is better)

### Quality Gates

- **V-score threshold**: ≥70 for inclusion
- **P₀ gating**: ≥0.20 baseline probability
- **Bin balancing**: Even coverage across pressure range [0.2, 1.0]

---

## Analysis Components

### Key Metrics

| Metric | Definition |
|--------|------------|
| **Violation Rate** | Fraction of samples containing forbidden word |
| **Suppression (ΔP)** | P₀ - P₁ (reduction due to constraint) |
| **Priming Index (PI)** | TMF - NF (attention to target vs. negation) |
| **FFN Contribution** | Change in target probability from FFN at each layer |
| **Patching Effect** | ΔP from replacing activations with baseline |

### Output Files

| File | Contents |
|------|----------|
| `completions_greedy.jsonl` | Greedy outputs with hidden states |
| `completions_samples.jsonl` | 16 stochastic samples per prompt |
| `detection_mapping.jsonl` | Violation detection results |
| `attention_metrics.csv` | Per-prompt attention metrics |
| `logit_lens.csv` | Layer-wise target probabilities |
| `ffn_attn_decomp.csv` | Component contributions by layer |
| `patching_results.csv` | Activation patching effects |

---

## Hardware Requirements

| Stage | Hardware | Time Estimate |
|-------|----------|---------------|
| Prompt Generation | CPU | 2-4 hours |
| Prompt Validation | CPU (API calls) | 1-2 hours |
| P_sem Computation | GPU (16GB+) | 30-60 minutes |
| Mechanistic Runs | GPU (40GB+) | 2-3 hours |
| Behavioral Runs | GPU (40GB+) | 4-6 hours |
| Activation Patching | GPU (40GB+) | 6-8 hours |

**Recommended**: NVIDIA A100-40GB or A100-80GB

---

## Model

All experiments use **Qwen2.5-7B-Instruct** (AWQ quantized for memory efficiency).

Download from Hugging Face:
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir ./models/Qwen-2.5-7B-Instruct
```

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{forbidden-word-effect-2024,
  title={The Forbidden Word Effect},
  author={Shailesh},
  year={2024},
  url={https://github.com/gut-puncture/Semantic-Gravity-RP}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Qwen team for the base model
- The interpretability community for developing the logit lens and activation patching techniques
- OpenAI for GPT-5.2 used in prompt generation and validation
