# Semantic Gravity: Quantifying the Efficiency Gap in Negative Constraint Adherence

This repository contains the code and data for the research paper "Semantic Gravity: Quantifying the Efficiency Gap in Negative Constraint Adherence."

## Overview

This project investigates how language models handle negative constraints (e.g., "don't use the word X") and introduces the concept of **Semantic Gravity** - the hypothesis that constraint adherence difficulty is functionally dependent on the contextual probability of the forbidden token.

## Repository Structure

```
.
├── main.ipynb                 # Main experiment notebook (run on GPU)
├── prompts.csv                # Dataset of 500 diverse prompts
├── research_methodology.md    # Research methodology and experimental design
├── implementation_plan.md     # Technical implementation details
├── scripts/                   # Prompt generation pipeline
│   ├── generate_prompts_batch.py
│   ├── check_batch_status.py
│   ├── deduplicate_prompts.py
│   └── README.md
└── venv/                      # Python virtual environment (gitignored)
```

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (for Qwen model)
- OpenAI API key (for GPT-5 experiments)

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install transformers accelerate bitsandbytes scipy pandas matplotlib openai sentence-transformers scikit-learn
```

### Running the Experiment

1. **Load the dataset**: The `prompts.csv` file contains 500 pre-generated, semantically diverse prompts.

2. **Execute the notebook**: Open `main.ipynb` in Jupyter or Google Colab (with GPU runtime).

3. **View results**: The notebook will generate:
   - `prompts_with_psem.csv` - Prompts with semantic pressure scores
   - `results_*.csv` - Results for each GPT-5 model
   - `collapse_curves.png` - Visualization of failure rates
   - `experiment_results.zip` - All results packaged for download

## Experiments

### Experiment 1: The Collapse Curve
- **Model**: Qwen-2.5-7B-Instruct (white box)
- **Goal**: Measure semantic pressure (P_sem) and failure rates
- **Output**: Collapse threshold identification

### Experiment 2: The Efficiency Gap
- **Models**: GPT-5-nano, GPT-5-mini, GPT-5-base (black box)
- **Goal**: Compare constraint adherence across model sizes
- **Hypothesis**: Larger models show better inhibition

## Dataset

The dataset consists of 500 prompts across 5 categories:
- **A_Idioms** (100): Fixed idioms with high probability completions
- **B_Facts** (100): Historical/scientific facts with medium-high probability
- **C_CommonSense** (100): Questions with 2-3 likely answers
- **D_Creative** (100): Open-ended story prompts with low probability
- **E_OOD** (100): Counter-intuitive scenarios (out-of-distribution)

All prompts were generated using GPT-4o and semantically deduplicated using sentence embeddings (cosine similarity < 0.85).

## Regenerating Prompts

See `scripts/README.md` for instructions on regenerating the prompt dataset.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{semantic_gravity_2025,
  title={Semantic Gravity: Quantifying the Efficiency Gap in Negative Constraint Adherence},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025}
}
```

## License

[Add your license here]

## Contact

[Add contact information]
