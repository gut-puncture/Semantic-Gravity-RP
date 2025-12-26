# Execution Plan for Semantic Gravity (Colab, A100)

This plan is exhaustive and unambiguous. It implements `specification.md`
and `implementation-plan.md` exactly. Do not change methodology.

## 0. Non-negotiables and order

0.1 Sources of truth:
- `specification.md`
- `implementation-plan.md`

0.2 If any instruction here conflicts with those files, stop and resolve by
deferring to `specification.md`.

0.3 Required execution order:
- Correctness engine (detection + token span mapping) and unit tests.
- Prompt pool creation and validation (prompts first).
- Pressure probe computation and gating/balancing.
- Main runs (greedy mechanistic + stochastic samples).
- Detection/mapping on completions and behavioral metrics.
- Mechanistic analyses.
- Activation patching.
- Bootstrap, figures, and paper artifacts.

0.4 Determinism rules:
- Greedy decoding only for mechanistic analyses.
- Stochastic sampling only for failure probability estimation.

0.5 Hard halt:
- If token span mapping errors > 0, stop immediately and fix before
continuing.

0.6 All output artifacts must be saved under one run root:
`outputs/experiment_run_YYYYMMDD_HHMM/`.

0.7 Every long loop must be resumable by reading existing output files and
skipping completed items.

0.8 A100 GPU is required in Colab; abort if not present.

0.9 All GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch requests/responses must be logged with request + response JSON; if logging fails, halt immediately.

0.10 Prompt list must be generated and finalized before any main model runs.

## 1. File layout and module map

1.1 Keep Python modules under `src/`.

1.2 Ensure these module files exist (create if missing):
- `src/config.py`
- `src/utils.py`
- `src/api_clients.py`
- `src/data_mining.py`
- `src/validator.py`
- `src/dataset_pipeline.py`
- `src/prompt_builder.py`
- `src/runner.py`
- `src/detector.py`
- `src/metrics_psem.py`
- `src/metrics_attn.py`
- `src/patching.py`
- `src/visualize.py`

1.3 Create a single end-to-end notebook:
- `notebooks/semantic_gravity.ipynb`
- This notebook must import the modules above and run the full pipeline in
  the exact order defined in Section 0.

1.4 All outputs are written to:
`outputs/experiment_run_YYYYMMDD_HHMM/` with the substructure in Section 13.

## 1.5 Current repo status (read before coding)

Status legend:
- Done: implemented and aligns with spec (keep as-is).
- Partial: implemented but must be updated to match spec.
- Missing: not implemented yet.

Module inventory:
- `src/config.py` (Partial)
  - Keep: `CONFIG`, `PROMPT_TEMPLATES`, `get_base_paths()`, `setup_directories()`,
    `validate_environment()`.
  - Update: Drive root naming to match Colab path policy, add explicit output
    subpaths required in Section 13, allow runtime model path override.
- `src/utils.py` (Partial)
  - Keep: `normalize_for_match()` (matches spec), `ModelWrapper` singleton,
    `compute_token_char_spans()`, `map_word_to_tokens()`.
  - Update: add `normalize_for_word_match()` alias or rename; add stricter
    token span verification and mapping error handling in a dedicated
    `src/detector.py` (utils has no hard-fail logic).
- `src/api_clients.py` (Partial)
  - Keep: GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch client, Wikidata client, ConceptNet client (expected unused), idiom CSV cache loader.
  - Update: log full request/response JSON to disk; explicit GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) model
    selection; stronger retry/backoff + rate limiting policy.
- `src/data_mining.py` (Partial)
  - Keep: CandidatePrompt dataclass; idioms/facts/common-sense generators;
    creative and OOD generators as base.
  - Update: enforce "prompt must not contain X"; ensure exact style IDs per spec;
    generate OOD 1000 candidates before validation; allow
    `OpenAIFallbackGenerator` for any category when a primary source is short.
- `src/validator.py` (Partial)
  - Keep: ValidationResult, PromptValidator, TargetTracker, PromptSelector.
  - Update: use `prompt_builder` for baseline/negative formatting; replace
    `compute_semantic_pressure()` with `metrics_psem` algorithm (P0 and P1);
    ensure strict JSON logging and acceptance rules.
- `src/dataset_pipeline.py` (Partial)
  - Keep: overall scaffold for dataset build.
  - Update: pressure gating with tau adjustment, bin balancing, prompt metadata
    export (`prompts.csv` + `prompts_metadata.json`), manual spot check hooks.
- `src/prompt_builder.py` (Missing)
  - Implement exact baseline + negative templates and `build_prompt()`.
- `src/detector.py` (Missing)
  - Implement correctness engine with strict mapping + hard halt.
- `src/runner.py` (Missing)
  - Implement greedy + stochastic runs and save traces/completions.
- `src/metrics_psem.py` (Missing)
  - Implement P_sem per spec (variant enumeration + teacher forcing).
- `src/metrics_attn.py` (Missing)
  - Implement attention metrics and span mapping.
- `src/patching.py` (Missing)
  - Implement activation patching.
- `src/visualize.py` (Missing)
  - Implement plots and tables.
- `notebooks/semantic_gravity.ipynb` (Missing)
  - Create from scratch; `notebooks/01_inference_template.ipynb` is only a stub.

## 2. Colab bootstrap (must be the first notebook cell block)

2.1 Create a `RUN_ID` string:
- Format: `experiment_run_YYYYMMDD_HHMM` (local time).
- Example: `experiment_run_20250125_1542`.
- Set `OUTPUT_ROOT = "/content/drive/MyDrive/semantic_gravity/outputs/" +
  RUN_ID`.

2.2 Mount Google Drive:
- `from google.colab import drive`
- `drive.mount("/content/drive", force_remount=False)`
- If mount fails, raise a hard error and stop.

2.3 Validate GPU:
- Run `!nvidia-smi -L`.
- Parse output and confirm "A100".
- If not A100, raise a hard error and stop.

2.4 Environment flags for efficiency:
- `os.environ["TOKENIZERS_PARALLELISM"] = "false"`
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`

2.5 Package versions:
- Import all required packages.
- If `transformers.__version__ < 4.37`, install a newer version.
- Log exact versions for all packages in `run_metadata.json`.
- Do not change versions after logging.

2.6 Save runtime metadata to:
- `outputs/.../run_metadata.json`
- Include: timestamp, GPU name, CUDA version, package versions, repo git
  commit hash if available, model path, model id, tokenizer vocab size.

## 3. Config and utilities (module 1)

3.1 Update existing `src/config.py` to define or expose:
- A stable path API (either explicit keys in `CONFIG` or derived via
  `get_base_paths()` + `setup_directories()`), covering:
  - Drive root for data + outputs
  - data/{raw,candidates,validated}
  - outputs/experiment_run_*/{runs,figures,appendix_examples,errors}
- `SEEDS = {"python": 42, "numpy": 42, "torch": 42}`
- `MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"`
- `MODEL_PATH`: Drive path provided at runtime (must exist); allow override via
  env var or notebook input.
- `PROMPT_TEMPLATES` dict with exact strings from spec.
  - This already exists; verify exact string matches and keep.

3.2 `setup_directories()` (already exists; update as needed):
- `os.makedirs(..., exist_ok=True)` for all required folders.
- If Drive is not mounted, log a warning (do not crash).
- Return a dict of created paths (current behavior) and ensure it includes all
  required output subpaths.

3.3 Update existing `src/utils.py` to define or expose:
- `normalize_for_word_match(text)` (alias or rename from `normalize_for_match`):
  - Unicode NFKC
  - `.lower()`
  - replace punctuation with spaces
  - collapse whitespace to single spaces
  - `.strip()`
  - Punctuation set must include ASCII punctuation and also unicode dash
    characters `\u2013` and `\u2014`.
- `set_seed(seed)` for random, numpy, torch, torch.cuda.
- `ModelWrapper` singleton:
  - Loads tokenizer + model once.
  - Uses `device_map="auto"` and `torch_dtype=torch.bfloat16`.
  - `model.eval()` and `torch.set_grad_enabled(False)`.
  - Performs `transformers` version check (`>=4.37`).
- Logging helper:
  - Keep `setup_logging()`; add `get_logger()` only if needed by new modules.

3.4 Acceptance criteria:
- `setup_directories()` creates all required directories.
- `ModelWrapper` loads without reloading on repeated calls.
- `normalize_for_word_match` passes unit tests in Section 4.

## 4. Correctness engine (module 5, must be implemented and tested first)

4.1 Create `src/detector.py` with:
- `word_present(target, completion)` returning bool; use decoded string for detection.
- `token_char_spans(tokenizer, ids)` using prefix incremental decode.
- `find_word_spans(decoded, target)` using case-insensitive match with
  non-alphanumeric boundaries (per `str.isalnum()`).
- `map_word_to_tokens(...)` that maps regex spans to token spans.
- `detect_and_map(...)` that ties it together and returns:
  - `word_present`
  - `token_spans` (list of [start,end] token indices)
  - `mapping_error` bool

4.2 Prefix incremental decode (only allowed method):
- For i in 1..N:
  - `s_i = tokenizer.decode(ids[:i], clean_up_tokenization_spaces=False,
    skip_special_tokens=True)`
  - `end_i = len(s_i)`
- Token i span = `(end_{i-1}, end_i)` with `end_0 = 0`.

4.3 Mapping logic:
- Find all regex matches in decoded string.
- For each match [a,b):
  - Find smallest contiguous token span whose char spans cover [a,b).
  - Decode that token span and verify it matches the substring.
  - Verify left and right boundaries are non-alphanumeric.
  - If failed, expand window by +/-1 token and retry.
  - If still failed, brute force all contiguous spans up to length 8.
- If mapping fails and `word_present == True`:
  - `mapping_error = True`
  - Save to `errors/mapping_errors.jsonl` and increment counter.

4.4 Unit tests (must run before anything else):
- "space" does not match "spacetime".
- "space." and "space," are detected.
- "space" does not match "space2".
- "space" matches "space-time".
- " Space" detected.
- Multi-token split mapping (use tokenizer on a word known to split).
- All tests must pass or halt.

4.5 Hard halt condition:
- If mapping error count > 0 OR mapping error rate > 0.1%, stop run.

## 5. Prompt pool creation and validation (prompts first)

5.1 Update existing `src/api_clients.py`:
- Use GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch-only client (OpenAI Batch API).
  - Build JSONL requests, upload, create batch, poll, download output.
  - Log request/response JSON for every item; if logging fails, halt.
  - Model `gpt-5.2-2025-12-11` with text.format `json_object`,
    `reasoning.effort="none"`, `reasoning.summary="auto"`, `text.verbosity="low"`, `store=true`.
  - Use max_output_tokens=2000 for JSON-mode calls.
  - Include the word "json" in the prompt and a JSON example for JSON-mode calls.
  - Parse JSON only from response output text; if empty/malformed, re-queue a corrected batch item.
- `Wikidata` client using `SPARQLWrapper` with exact queries in spec.
- `ConceptNet` client for `api.conceptnet.io` edges.

5.2 Update existing `src/data_mining.py` generators for each category:
- A Idioms: use existing `data/raw/idioms.csv` (cache), filter:
  - at least 3 tokens
  - final token alphabetic after stripping punctuation
  - reject malformed strings
  - form prompt with last word replaced by "____"
  - style ids: I1, I2, I3
  - if fewer than 1,000 candidates, use GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch fallback (idiom_full/target_word) and format with I1–I3
- B Facts: run SPARQL for capitals and currencies, filter answers to
  `^[A-Za-z]+$`, style ids F1, F2, F3.
  - if fewer than 1,000 candidates, use GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch fallback (subject/relation/target_word) and format with F1–F3
- C Common sense: ConceptNet is expected to be down. Generate the full pool with GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch fallback:
  - Return subject, relation (UsedFor/MadeOf/HasProperty), target_word
  - Build question_text using C1-C3 templates
  - Enforce diversity (no duplicate subject/target, balance relation types)
  - Target 1,000 candidates (prompts_per_category * 2)
- D Creative: select target X from wordfreq (Zipf 3.5-6.0), seed from `data/raw/writingprompts.txt`, generate K=3
  micro-story prompts via GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch (must not contain X), fixed format.
  - Target 1,800 candidates total (K=3 per target across 600 targets).
  - If WritingPrompts is unavailable, proceed without scenario seeds and use the fallback schema.
- E OOD: generate 1,800 candidates total (K=3 per target across 600 targets),
  then select 500 after validation and pressure ranking.
  - If bins are short after gating, fill with next-highest S(p) prompts (no regeneration).

5.3 Update existing `src/validator.py`:
- Apply deterministic filters before any model validation:
  - Reject if target appears in prompt by whole-word match (case-insensitive).
  - Reject if target appears after stripping non-letters from prompt.
  - Reject if creative is not exactly two sentences.
  - Reject if out-of-distribution is not one or two sentences.
  - Reject if target is not strictly alphabetic.
- Call GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) (batch-only) to return strict JSON:
  - is_one_word_answer_enforced
  - best_one_word_answer
  - top3_one_word_answers
  - is_X_best
  - ambiguity_score
  - leaks_answer
  - naturalness_score
  - comments
- Apply acceptance rules from spec.
- Store raw validation JSON in `data/gpt5_validation.jsonl`.

5.4 Candidate selection (D/E):
- After validation, select the best accepted prompt per target (600 targets -> 600 prompts), then apply P_sem gating and bin balancing.
- If bins are short or repetition caps reduce counts, fill to 500 with the next-highest S(p) prompts from the remaining validated pool (no regeneration).
- Compute S(p) = V(p) + 100 * P(p) for ranking within bins.

5.5 Prompt constraints:
- Every prompt must include "Answer with exactly one English word."
- Prompts must not contain X (case-insensitive, after normalization).
- If a candidate violates, discard and continue (no regeneration).

5.6 Diversity and edge case coverage (must be explicit):
- Compute distributions of:
  - prompt length in tokens
  - target word length in characters
  - target word initial letter
  - punctuation presence in question text
- Ensure each category includes:
  - at least 50 prompts with question length > 20 tokens
  - at least 50 prompts with question length < 8 tokens
  - at least 20 prompts where the question contains punctuation
- If any requirement fails, log shortfalls and continue (no regeneration).

5.7 Repetition control across categories:
- `target_word_normalized` may appear in at most 2 categories.
- If cap violated, discard candidate.
- If the category drops below 500 after repetition filtering, continue with remaining prompts and log the shortfall (no regeneration).

5.8 Manual spot check (mandatory):
- Sample 50 accepted prompts per category.
- If >10% fail acceptance criteria, tighten filters for future runs; do not regenerate in this run.

5.9 Save final prompts:
- `data/prompts.csv` with schema in Section 13.
- `data/prompts_metadata.json` with dataset-level metadata.
- `data/prompts.jsonl` (one JSON per prompt) for easy streaming.
- Stop here and review prompts before any model runs.

5.10 Update `src/dataset_pipeline.py` to orchestrate:
- candidate generation -> validation -> P0 scoring -> gating -> bin balancing
- export all files from 5.9
- GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch fallback is allowed for any category when a primary source yields fewer than 1,000 candidates
- If fallback still yields fewer than the target pool, proceed with the shortfall and log it (no regeneration)
- If validator acceptance yields fewer than 500 prompts in any category, proceed with the accepted pool and log the shortfall (no regeneration)

## 6. Pressure probe and gating (module 5)

6.1 Create `src/metrics_psem.py`:
- `enumerate_surface_variants(X)`:
  - case: X, lower, title, upper
  - whitespace: "", leading space, trailing space, both
  - punctuation: trailing period at minimum (optional comma, ?, ! only if
    logged)
- `token_sequences_for_variants(...)`:
  - Tokenize each variant (no special tokens).
  - Decode and normalize; keep only those matching X_norm.
  - Deduplicate identical token sequences.

6.2 `compute_p_sem(model, tokenizer, context_ids, token_sequences)`:
- Use teacher forcing with `past_key_values`.
- P(s|context) = product of per-token probabilities.
- P_sem = sum over all sequences.
- Use `torch.inference_mode()` and bfloat16.

6.3 Compute per prompt:
- P0 = P_sem for baseline prompt.
- P1 = P_sem for negative prompt.
- Delta, Relative, Log suppression metrics.

6.4 Pressure gating:
- Start tau = 0.20.
- If >500 prompts per category after gating, keep 500 by bin balancing.
- If <500, lower tau by 0.05 and recompute until 500 or tau == 0.05.
- Log final tau for each category.
- If any category still has <500 after tau lowering, fill with next-highest S(p) prompts and log bin shortfalls (no regeneration).

6.5 Bin balancing:
- Bins: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0].
- Target 100 prompts per bin per category.
- If a bin has >100, sample with fixed seed.
- If a bin has <100, keep all and record shortfall in metadata.

6.6 Save:
- `runs/psem.csv`
- `runs/pressure_bins.csv`

## 7. Prompt formatting and model runner (module 3)

7.1 Create `src/prompt_builder.py`:
- Baseline template (exact):
  - "Answer with exactly one English word.\nQuestion: {q}\nAnswer:"
- Negative template (exact):
  - "Answer with exactly one English word.\nDo not use the word \"{x}\" anywhere in your answer.\nQuestion: {q}\nAnswer:"
- Provide `build_prompt(question, target, condition)` to return exact string.
- Unit test: verify exact string matches expected output.

7.2 Create `src/runner.py`:
- Load model/tokenizer via `ModelWrapper`.
- For each prompt in `data/prompts.csv`:
  - Build baseline and negative prompts.
  - Mechanistic run: greedy decode
    - `max_new_tokens=8`, `do_sample=False`
    - `output_hidden_states=True`, `output_attentions=True`
    - `return_dict_in_generate=True`
    - Save full internal states for generated tokens
  - Behavioral run: stochastic samples
    - `temperature=1.0`, `top_p=0.9`
    - `max_new_tokens=10`, `num_return_sequences=16`
  - Save outputs after each prompt to allow resume.

7.3 Storage:
- Mechanistic traces:
  - `runs/mechanistic_trace/{prompt_id}_{condition}.pt`
  - Include: input_ids, generated_ids, hidden_states, attentions.
- Greedy completions:
  - `runs/completions_greedy.jsonl`
- Sampled completions:
  - `runs/completions_samples.jsonl`

7.4 Memory management:
- After each prompt: `del` large tensors and `torch.cuda.empty_cache()`.
- Move tensors to CPU before saving to reduce GPU memory pressure.

## 8. Detection and behavioral metrics (module 4)

8.1 Run detection/mapping on:
- `runs/completions_greedy.jsonl`
- `runs/completions_samples.jsonl`

8.2 Save `runs/detection_mapping.jsonl` with:
- prompt_id, condition, completion_text, completion_norm
- target_word, target_word_norm
- word_present, token_spans, mapping_error
- format_adherence (one-word after normalization)

8.3 Compute behavioral metrics per prompt (negative condition):
- violation_rate (fraction of 16 samples with word_present)
- format_adherence_rate
- clean_success_rate
- Save to `runs/behavior_metrics.csv`.

8.4 Enforce hard halt if mapping errors are detected.

## 9. Mechanistic measurements (module 5)

9.1 Prompt span tokenization:
- For each prompt, find char spans for:
  - instruction sentence
  - negation phrase ("Do not use")
  - target mention inside instruction
  - question span
- Use `tokenizer(..., return_offsets_mapping=True)` and map spans to token ids.

9.2 Attention metrics (`src/metrics_attn.py`):
- Use mechanistic traces and attention weights.
- For generated position p (first answer token):
  - Mass(S) = sum attention over context positions in span S.
- Compute per layer and per head:
  - IAR, NF, TMF, PI
- Save to `runs/attention_metrics.csv` with layer/head granularity and
  per-prompt aggregate.

9.3 Layerwise logit lens:
- For each layer, project hidden state at answer position through
  `model.model.norm` then `lm_head`.
- Compute probability mass over first tokens of sequences in S(X).
- Save to `runs/logit_lens.csv`.

9.4 Attention vs FFN contribution decomposition:
- Hook each layer to capture:
  - h_in
  - attn_out
  - ffn_out
- Compute proxy probabilities at:
  - h_in
  - h_in + attn_out
  - h_in + attn_out + ffn_out
- Save to `runs/ffn_attn_decomp.csv`.

## 10. Activation patching (module 5, causal check)

10.1 Subset selection:
- 5 pressure bins based on P0.
- Outcome under negative instruction (success vs failure).
- Target 50 prompts per bin (250 total).
- Ensure at least 30 failures overall; oversample bins with failures if needed.
- Save subset list to `runs/patching_subset.json`.

10.2 Patching procedure:
- For each prompt in subset:
  - Run baseline and negative forward passes (greedy, no sampling).
  - Cache residual stream, attention outputs, and key/value tensors.
  - Apply patch types A-D as specified in `specification.md`.
  - Compute P_sem and greedy next token after patch.

10.3 Save results:
- `runs/patching_results.csv` with:
  - prompt_id, category, bin, outcome, layer, patch_type
  - P_sem_original_neg, P_sem_patched, delta_P, flip_indicator
  - greedy_token_original, greedy_token_patched

## 11. Bootstrap (module 5)

11.1 Bootstrap across prompts:
- B = 1000 iterations.
- Resample prompts with replacement (choose global or per-category and stick
  with it).

11.2 Metrics with 95% CI:
- violation rate per pressure bin
- suppression metrics per bin
- attention metrics per bin
- patching effect sizes per bin

11.3 Save to `runs/bootstrap_results.csv`.

## 12. Visualization and paper artifacts (module 6)

12.1 Create figures in `figures/`:
1) Violation rate vs P0 bin with CI.
2) Relative suppression vs P0 with CI.
3) Attention routing metrics vs P0, success vs failure.
4) Priming Index PI vs violation probability.
5) Layerwise logit lens curves baseline vs negative, success vs failure.
6) Attn vs FFN contributions by layer.
7) Activation patching effects by layer and bin.

12.2 Save quantitative tables:
- `tables.json` with exact values used in the Results section.

12.3 Qualitative examples:
- Save 20 curated failures to `appendix_examples/*.json` with:
  - prompt, X, output, token span mapping, attention metrics summary.

## 13. Output file schemas (must be explicit and stable)

13.1 `data/prompts.csv` columns:

**Note**: prompt_id is a string key of the form '{category}_{target_word_normalized}_{hash8}'
where hash8 is the first 8 characters of SHA256(question_text||target_word||category).
This ensures uniqueness across prompts with the same target word. Treat it as an opaque
identifier; do not assume any specific format or attempt to parse it.

- prompt_id (string key: '{category}_{target_word_normalized}_{hash8}')
- question_text (string; raw cloze question without instruction)
- prompt_text (string; full baseline prompt with instruction)
- category (idioms|facts|common_sense|creative|ood)
- target_word (string)
- target_word_normalized (string)
- prompt_style_id (string)
- source_trace (string)
- validation_json_ref (string; ref to validated JSONL, e.g. 'facts_validated.jsonl#prompt_id=facts_paris')
- p0 (float; baseline P_sem)
- p1 (float; negative P_sem)
- p0_bin (string like '0.2-0.4')
- v_score (int; validation score)
- p_sem (float; alias of p0 for backward compatibility)
- s_score (float; combined selection score)

13.2 `data/prompts_metadata.json`:
- prompt_id_format: "{category}_{target_word_normalized}_{hash8}"
- counts by category and bin
- tau per category (actual values from tau-lowering loop)
- bin_shortfalls: per category dict of bin -> shortfall count
- gating_failures: list of categories that failed to reach 500
- sampling seeds
- GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) model ids, batch ids, and timestamps
- dataset hash (SHA256 of prompts.csv)

13.3 `data/gpt5_validation.jsonl`:
- one JSON per candidate: prompt_id, category, candidate_text, target_word,
  raw_response_json, parsed_fields, acceptance_decision
- also write `data/gpt5_requests.jsonl` for all GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch requests/responses

13.4 `runs/completions_greedy.jsonl`:
- prompt_id (string key), condition, prompt_text
- generated_text, generated_token_ids
- finish_reason, timestamp

13.5 `runs/completions_samples.jsonl`:
- prompt_id (string key), condition, sample_id (0..15)
- generated_text, generated_token_ids
- finish_reason, timestamp

13.6 `runs/detection_mapping.jsonl`:
- prompt_id, condition, completion_text, completion_norm
- target_word, target_word_norm
- word_present (bool)
- token_spans (list of [start,end])
- mapping_error (bool)
- format_adherence (bool)

13.7 `runs/psem.csv`:
- prompt_id, category, p0, p1
- delta, relative, log
- num_token_sequences

13.8 `runs/pressure_bins.csv`:
- prompt_id, category, p0, bin_id, bin_lower, bin_upper

13.9 `runs/attention_metrics.csv`:
- prompt_id, condition, layer, head
- iar, nf, tmf, pi
- aggregate_flag (per-head or per-layer aggregate)

13.10 `runs/logit_lens.csv`:
- prompt_id, condition, layer
- p_sem_first_token
- greedy_token, greedy_token_prob

13.11 `runs/ffn_attn_decomp.csv`:
- prompt_id, condition, layer
- p_h_in, p_h_in_plus_attn, p_h_out
- attn_contrib, ffn_contrib

13.12 `runs/patching_results.csv`:
- prompt_id, category, p0_bin, outcome
- layer, patch_type
- p_sem_original_neg, p_sem_patched, delta_p
- flip_indicator, greedy_token_original, greedy_token_patched

13.13 `runs/bootstrap_results.csv`:
- metric_name, group_key, mean, ci_low, ci_high, n_prompts, seed

13.14 `errors/mapping_errors.jsonl`:
- prompt_id, condition, completion_text, target_word
- generated_token_ids, decoded_text, error_reason

## 14. Testing and verification checklist (must be executed)

14.1 Unit tests (before anything else):
- `detector.py` tests pass.
- `prompt_builder.py` template tests pass.
- `metrics_psem.py` sanity check: P_sem in [0,1].

14.2 Mini integration test (before full run):
- Run the full pipeline on 2 prompts per category.
- Confirm all output files are created and non-empty.
- Confirm no mapping errors.

14.3 Full run checks:
- After each major phase, validate:
  - expected file counts
  - no missing categories or bins
  - outputs are readable and schema-valid

14.4 Hard halt conditions:
- mapping_error_count > 0
- API responses missing required fields
- prompts per category not equal to 500

## 15. Final run order (execute exactly)

**Note**: prompt_id is a string key of the form '{category}_{target_word_normalized}'
(e.g. 'facts_paris'). Treat it as an opaque identifier; do not assume numeric
format or attempt to cast to int.

15.1 Mount Drive, verify A100, log environment.
15.2 Run correctness engine tests and halt on failure.
15.3 Build candidate pools per category.
15.4 Validate candidates with GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch.
15.5 Compute P0 and gate/balance prompts.
15.6 Save prompt list and review (prompts first).
15.7 Run greedy paired runs and save traces.
15.8 Run 16 stochastic samples per condition.
15.9 Run detection/mapping on all completions.
15.10 Compute behavioral metrics.
15.11 Compute mechanistic metrics (attention, logit lens, decomp).
15.12 Run activation patching subset.
15.13 Run bootstrap.
15.14 Generate figures and tables.
15.15 Export appendix examples and paper artifacts.
