# Prompt Generation System

This directory contains scripts to generate high-quality, diverse prompts using OpenAI's Batch API and semantic deduplication.

## Overview

The system generates **600 raw prompts** (120 per category) using GPT-5, then deduplicates to **500 diverse prompts** (100 per category) using semantic embeddings.

## Prerequisites

```bash
pip install openai pandas sentence-transformers scikit-learn
```

## Workflow

### Step 1: Generate Prompts (Batch API)

```bash
python generate_prompts_batch.py
```

**What it does:**
- Creates batch requests for 600 prompts across 5 categories
- Submits to OpenAI Batch API using GPT-5
- Saves batch ID to `batch_id.txt`

**Output:** `batch_id.txt`, `batch_input.jsonl`

---

### Step 2: Monitor Batch Status

**Option A: Manual Check**
```bash
python check_batch_status.py
```

**Option B: Auto-Monitor** (recommended)
```bash
python check_batch_status.py --monitor
```

**What it does:**
- Checks batch status every 60 seconds
- Downloads results when complete
- Parses output and saves to `prompts_raw.csv`

**Output:** `batch_output.jsonl`, `prompts_raw.csv` (~600 prompts)

**Note:** Batches typically complete within 24 hours.

---

### Step 3: Deduplicate Prompts

```bash
python deduplicate_prompts.py
```

**What it does:**
- Computes semantic embeddings using SentenceTransformer
- Finds and removes near-duplicates (cosine similarity > 0.85)
- Selects 100 most diverse prompts per bucket
- Reports diversity metrics

**Output:** `prompts.csv` (~500 prompts)

---

### Step 4: Run Experiment

Open `main.ipynb` and execute cells. The notebook will load `prompts.csv` automatically.

## Categories

- **A_Idioms**: Fixed idioms (high probability)
- **B_Facts**: Historical/scientific facts (medium-high probability)
- **C_CommonSense**: Questions with 2-3 likely answers (medium probability)
- **D_Creative**: Open-ended story prompts (low probability)
- **E_OOD**: Counter-intuitive, surreal scenarios (variable probability)

## Files Generated

| File | Description | Size |
|------|-------------|------|
| `batch_id.txt` | Batch job ID | ~30 bytes |
| `batch_input.jsonl` | Batch API requests | ~50 KB |
| `batch_output.jsonl` | Raw API responses | ~200 KB |
| `prompts_raw.csv` | Raw prompts before dedup | ~600 rows |
| `prompts.csv` | **Final deduplicated prompts** | ~500 rows |

## Troubleshooting

**Batch API errors?**
- Check your OpenAI API key has Batch API access
- Verify you have sufficient credits

**Deduplication taking too long?**
- The SentenceTransformer model downloads on first run (~90 MB)
- Subsequent runs are fast

**Want different similarity threshold?**
- Edit `SIMILARITY_THRESHOLD` in `deduplicate_prompts.py` (default: 0.85)

## Configuration

Edit the following in `generate_prompts_batch.py`:
- `PROMPTS_PER_BUCKET`: Default 120
- `BATCH_SIZE`: Default 5 (prompts per API call)
- `OPENAI_API_KEY`: Update with your key
