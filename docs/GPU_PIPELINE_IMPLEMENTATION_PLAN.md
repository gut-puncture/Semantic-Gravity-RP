# Semantic Gravity GPU Pipeline Updates Plan

## Context and Goal
The current Colab run took hours with low GPU utilization. The main bottlenecks
are sequential P_sem/P1 computation and per-prompt model calls in later stages.
We are moving to a rented GPU, so we must:
- Batch model work to keep VRAM utilization high (leave ~5-10 GB free).
- Add logging and progress visibility for all long-running steps.
- Add checkpointing and resume across the pipeline to avoid rework.
- Keep outputs correct and reproducible.

Local smoke tests are not possible on the laptop, so correctness validation
will be done on the Prime Intellect GPU with a small sample.

## Decisions
- Filtered validated prompts will be written to a new folder:
  `data/validated_single_token/` (keeps original validated files intact).
- Auto batch-size fallback on OOM is enabled and logged (no silent changes).
- A small GPU-side sanity check will be run before the full job.

## Scope (Notebook Cells + Code)
Notebook cells that run the model and need batching/logging/checkpoints:
- Model load (cell 12)
- Single-token filtering (cell 13)
- P_sem/P1 finalization (cell 16)
- Mechanistic + behavioral generation (cells 18, 20)
- Detection mapping (cell 22)
- Attention metrics + logit lens/decomp (cell 24)
- Activation patching (cell 26)

Code files to update:
- `src/metrics_psem.py`
- `src/validator.py`
- `src/dataset_pipeline.py`
- `src/runner.py`
- `src/metrics_attn.py`
- `src/patching.py`
- `notebooks/semantic_gravity.ipynb`

## Implementation Details (By Component)

### 1) Notebook Paths and Logging
- Replace Colab-only paths with VM-safe environment variables:
  `DATA_ROOT`, `MODEL_PATH`, `SRC_PATH`, `HF_HOME`.
- Write all logs to `OUTPUT_ROOT/logs/` and mirror to stdout.
- Replace the A100-only check with a generic GPU check + memory report.

### 2) Single-Token Filtering (Cell 13)
- Write filtered files to `data/validated_single_token/`.
- Add a small cache for `token_sequences_for_variants` to avoid redundant work.
- Add per-category progress logging and a summary JSON log.
- Add a checkpoint marker per category so the step can resume safely.

### 3) Batched P_sem and P1 (Finalization)
Core change: replace per-prompt sequential probability with a batched forward pass.

Approach (correct-by-construction):
- For each prompt, build context token IDs with the same `add_special_tokens`
  setting used elsewhere.
- For each target word, enumerate token sequences as usual.
- Convert each (context_ids, token_sequence) into a "task".
- Batch tasks by padding to a max length and using `attention_mask`.
- Compute log-probabilities using logits at positions
  `context_len - 1 + t` for token `t` in the sequence.
- Sum exp(log-prob) across sequences per prompt to produce P_sem.
- Clamp to [0, 1] and preserve strict error behavior.

Checkpointing:
- Write P_sem results to JSONL as they finish.
- On resume, load existing results and skip completed prompts.
- Same pattern for P1 (negative prompt) results.

### 4) Batched Generation (Mechanistic + Behavioral)
Mechanistic (greedy, full traces):
- Use microbatches (small batch size) due to hidden state/attention capture.
- Split outputs per prompt and save per-prompt `.pt` traces as today.

Behavioral (sampling):
- Batch prompts for generation (num_return_sequences per prompt).
- Append results to JSONL immediately for resume safety.

Logging:
- Log throughput (prompts/sec) and ETA every N prompts.
- Log GPU memory usage periodically.

### 5) Detection Mapping (Cell 22)
- Stream input JSONL and write output JSONL incrementally.
- Track processed `prompt_id|condition|sample_id` keys for resume.
- Log progress and mapping error counts as the run proceeds.

### 6) Attention Metrics + Logit Lens / Decomp
- Batch forward passes with padded inputs and attention masks.
- Write results row-by-row to CSV (append mode).
- Resume by skipping rows already present in output CSV.

### 7) Activation Patching
- Keep conservative microbatching (batch=1 unless shapes allow batching).
- Write partial results incrementally and resume by skipping existing rows.

## Checkpoint / Output Locations
- Filtered validated prompts: `data/validated_single_token/*.jsonl`
- P_sem checkpoint: `data/validated_single_token/psem_checkpoint.jsonl`
- P1 checkpoint: `data/validated_single_token/p1_checkpoint.jsonl`
- Run logs: `outputs/<run_id>/logs/*.log`
- Partial CSVs/JSONL: `outputs/<run_id>/runs/*_partial.*`

## Task Checklist
- [x] Update notebook paths and logging config in `notebooks/semantic_gravity.ipynb`
- [x] Update cell 13 to write to `data/validated_single_token/` with resume support
- [x] Add batched P_sem computation in `src/metrics_psem.py`
- [x] Add checkpointed, batched P_sem in `src/validator.py`
- [x] Add checkpointed, batched P1 computation in `src/dataset_pipeline.py`
- [x] Batch mechanistic + behavioral generation in `src/runner.py`
- [x] Finalize detection mapping streaming (include `sample_id` in output rows for resume)
- [x] Batch logit lens/decomp and add resume in `src/metrics_attn.py`
- [x] Add microbatch + resume in `src/patching.py`
- [x] Add resumable streaming attention metrics in `src/metrics_attn.py`
- [x] Fix missing `time` import in `compute_logit_lens_and_decomp`
- [x] Update `notebooks/semantic_gravity.ipynb` to use new batch params + paths
- [ ] Prime Intellect runbook: VM setup, model download, full run, sync outputs back
- [ ] GPU sanity check: compare batched vs sequential on a small sample
