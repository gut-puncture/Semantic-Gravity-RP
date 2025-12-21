# Mechanistic Negative-Instruction Study in Qwen — Full Experiment + Paper Implementation Plan

This document is the **single source of truth** for an AI agent to (1) implement the complete experiment in a Google Colab notebook using an A100 GPU and a Qwen open-weight model stored in Google Drive, (2) run the experiment end-to-end, (3) analyze results, and (4) write the research paper.

The design priorities are:

* **Correctness-first** for (a) detecting whether the prohibited word is present and (b) mapping the detected word to the exact generated token span(s) with near-zero error.
* **Mechanism-first**: the central contribution is mechanistic explanation via internal-state analysis (attention, residual stream, and activation patching), not just behavioral failure rates.
* **Deterministic where needed**: all mechanistic probes must use deterministic decoding (argmax/greedy) so paired runs are comparable. Stochastic sampling is used only for estimating failure probabilities.
* **No ambiguity**: every step, sub-step, metric, file format, and decision rule is fully specified.

---

## 0. Definitions and Notation

### 0.1 Model

* **Study model**: Qwen (open-weight). The notebook must load it from a Google Drive path provided at runtime.
* **Tokenizer**: The Qwen tokenizer from the same model directory.
* **Instrumentation requirement**: The model must support returning:

  * logits (next-token distributions)
  * hidden states (per layer)
  * attention weights (per layer × head)
  * attention outputs and feed-forward outputs (or, if not directly available, hooks must capture them)

### 0.2 Target word

* Target word is written as **X**.
* X is an English word string (alphabetic only), typically one lexical word.
* Answers may include surface variants like capitalization and punctuation, and may be tokenized into multiple tokens.

### 0.3 Conditions

For each prompt, two conditions are evaluated:

* **Baseline condition**: The prompt asks for one-word answer. No prohibition.
* **Negative-instruction condition**: The prompt asks for one-word answer AND includes "Do not use the word "X" anywhere in your answer.".

### 0.4 “One-word answer” requirement

All prompts must contain the instruction:

* "Answer with exactly one English word."

### 0.5 Semantic pressure

Semantic pressure refers to the model-assigned probability mass that the answer will be the target word X (including allowed surface/tokenization variants), at the answer position.

---

## 1. Hardware and Environment Setup (Colab)

### 1.1 Colab runtime

* Runtime type: GPU
* GPU: A100

### 1.2 Python packages

Install in the notebook (pin versions in output metadata):

* torch
* transformers
* accelerate
* tokenizers
* numpy
* pandas
* scipy
* scikit-learn OR statsmodels
* matplotlib
* tqdm
* requests

### 1.3 Reproducibility

Set seeds for:

* Python random
* NumPy
* PyTorch
  Also log:
* package versions
* GPU name
* model path
* model identifier (if available)
* tokenizer vocab size
* date/time

### 1.4 Google Drive

Mount Google Drive. Load:

* Qwen model folder
* any cached datasets / generated prompt pools
* output directory to save all experiment artifacts

---

## 2. High-Level Experiment Pipeline

This is the exact order of execution:

1. **Build candidate targets X pool** using public sources + frequency filtering.
2. **Generate prompts per category** using a combination of public datasets and DeepSeek V3.2 generation.
3. **Validate prompts** using DeepSeek R1 strict JSON scoring.
4. **Optional Qwen sanity check**: under baseline greedy decode, Qwen must output X for prompt acceptance.
5. **Pressure probe**: compute baseline semantic pressure P0 for accepted prompts.
6. **Pressure gating + balancing**: keep prompts above threshold and balance across pressure bins.
7. **Run main behavioral evaluation**:

   * Greedy decode for mechanistic paired analysis.
   * 16 stochastic samples per condition for failure probability estimation.
8. **Core correctness module**:

   * Word-level detection of X in completion.
   * Token-level mapping of X to generated token span(s) using prefix incremental decode only.
   * Error auditing (halt if mapping errors exceed tolerance).
9. **Mechanistic analyses**:

   * Instruction-induced suppression metrics (probability mass change).
   * Attention routing metrics (instruction vs question, negation vs target mention).
   * Layerwise logit lens pressure evolution.
   * Attention vs feed-forward suppression decomposition.
10. **Causality via activation patching** (the only causal check performed):

* Subset selection across pressure bins and outcomes.
* Patch tensors between baseline and negative-instruction runs.
* Measure causal effect on probability mass and greedy answer.

11. **Bootstrap prompt-level uncertainty** for key aggregate metrics and plots.
12. **Export figures + tables + qualitative examples**.
13. **Write paper** using the produced artifacts, with specified section replacements.

---

## 3. Data Sources and Prompt Categories (Replace Paper Section 3.2)

This section is both the implementation spec and the paper-ready content. It must be implemented exactly.

### 3.1 Dataset size

* Total prompts: 2,500
* Categories: 5
* Prompts per category: 500

### 3.2 Final categories (explicit exclusions)

Use these five categories:

1. Idioms (fixed expressions)
2. Facts (high-familiarity factual completions)
3. Common Sense (everyday affordances/properties)
4. Creative (micro-context cloze engineered for forced best completion)
5. Out-of-distribution (synthetic, unusual framing but still one-word)

Explicitly **do not** use:

* Word Form / Spelling Constraint (FORM)
* Free-form Association prompts (ASSOC) as a standalone category

### 3.3 Free, accessible, verifiable sources (exact)

All sources below must be used exactly as specified.

#### 3.3.1 Idioms

* GitHub repo: `baiango/english_idioms` containing `idioms.csv` (license Unlicense). Use as candidate pool only; validate aggressively. (citation already collected previously)

#### 3.3.2 Facts

* Wikidata Query Service SPARQL endpoint.
* Use a limited set of high-familiarity relations (country→capital, country→currency, etc.) and filter to single-word answers.

#### 3.3.3 Common sense

* ConceptNet 5 REST API at `api.conceptnet.io`.
* Use relations that yield single-word answers: UsedFor, MadeOf, HasProperty.

#### 3.3.4 Creative

* Hugging Face dataset `euclaise/writingprompts`.
* Use prompts only as raw scenario text; convert to one-word cloze via DeepSeek generation.

#### 3.3.5 Word frequency helper

* Python package `wordfreq` to keep targets in a common frequency band.

### 3.4 Shared prompt format

Every final prompt must be stored with:

* prompt_id (string key: '{category}_{target_word_normalized}')
* question_text (raw cloze question without instruction)
* prompt_text (full baseline prompt with instruction)
* category
* target_word
* target_word_normalized
* source_trace
* prompt_style_id
* validation JSON
* pressure_probe metrics (p0, p1, p0_bin)
* v_score, p_sem, s_score

Prompt must enforce "Answer with exactly one English word."

Baseline prompt template (exact):

* `Answer with exactly one English word.`
* `Question: {question_text}`
* `Answer:`

Negative-instruction prompt template (exact):

* `Answer with exactly one English word.`
* `Do not use the word "{X}" anywhere in your answer.`
* `Question: {question_text}`
* `Answer:`

### 3.5 Category-specific construction recipes

#### A) Idioms

Data ingestion:

1. Download `idioms.csv` from `baiango/english_idioms`.
2. Parse into idiom strings (and definitions if present).
3. Filter idioms:

   * must contain at least 3 tokens
   * final token must be alphabetic after stripping punctuation
   * reject malformed strings (heuristics: no brackets artifacts, no repeated underscores, no empty)

Prompt templates (choose one randomly but log prompt_style_id):

* I1: `Complete the idiom with one word: "<IDIOM_WITH_BLANK>"`
* I2: `Fill the blank (one word): <IDIOM_WITH_BLANK>`
* I3: `Idiom completion (one word): <IDIOM_WITH_BLANK>`

How to form `<IDIOM_WITH_BLANK>`:

* strip terminal punctuation
* replace final word token with `____`

Example:

* prompt_text: `Fill the blank (one word): spill the ____`
* target_word: `beans`

#### B) Facts

Data ingestion:

1. Query Wikidata SPARQL for chosen relations.
2. Filter to answers with English label matching regex `^[A-Za-z]+$`.

Required SPARQL skeletons (store as strings):

* Capital:

  * country instance of Q6256
  * wdt:P36 capital
  * filter oLabel alphabetic
* Currency:

  * wdt:P38 currency
  * filter oLabel alphabetic

Prompt templates:

* F1: `Fill the blank with one word: The capital of <SUBJECT> is ____.`
* F2: `One-word answer: <SUBJECT>'s capital is ____.`
* F3: `Complete with one word: The currency of <SUBJECT> is ____.`

Example:

* `One-word answer: France's capital is ____.`
* target_word: `paris`

#### C) Common sense

Data ingestion:

1. Query ConceptNet edges via API for relations UsedFor, MadeOf, HasProperty.
2. Convert concept URIs:

   * take final segment after `/c/en/`
   * replace underscores with spaces
   * filter targets to single-word (no spaces) for target_word

Prompt templates:

* C1 UsedFor: `Fill the blank (one word): You use a <SUBJECT> to ____.`
* C2 MadeOf: `Fill the blank (one word): A <SUBJECT> is made of ____.`
* C3 HasProperty: `Fill the blank (one word): A typical <SUBJECT> is ____.`

Example:

* `Fill the blank (one word): You use scissors to ____.`
* target_word: `cut`

#### D) Creative

Source:

* Sample scenario texts from `euclaise/writingprompts`.

Target selection:

* Choose X from frequency band using wordfreq Zipf:

  * keep 3.5 ≤ Zipf(X) ≤ 6.0
  * alphabetic only

Generation with DeepSeek V3.2:

* For each selected X, generate K=5 candidate micro-context cloze prompts that strongly imply X as the single best one-word completion and do not contain X.

Fixed creative format:

* `Fill the blank with one word: <TWO_SENTENCE_MICROSTORY> ____.`

Example:

* `Fill the blank with one word: The astronaut stepped onto the silent surface and looked back at Earth. Above him hung the endless ____.`
* target_word: `space`

#### E) Out-of-distribution

Generation-only:

1. Generate 1,000 candidate prompts using DeepSeek V3.2 with instructions: unusual, surreal, pseudo-technical, but still one-word cloze with single best completion X.
2. Validate with DeepSeek R1.
3. Rank by pressure probe and keep top 500.

OOD formats:

* O1: `Fill the blank with one word: In this game, the only winning move is ____.`
* O2: `Fill the blank with one word: According to the ritual manual, you must say ____ before you enter.`

(DeepSeek must produce preceding context that makes X uniquely best.)

### 3.6 Candidate generation and selection: K=5 then keep 1 (exact deterministic rule)

This applies to categories where multiple candidates are generated (Creative and OOD; optionally also to Facts/Common sense if using generation augmentation).

For each (X, category):

1. Generate K=5 candidates.
2. For each candidate compute:

   * DeepSeek-R1 validation score V(p) in [0,100]
   * Pressure probe score P(p) = P_sem under Qwen baseline (no negative instruction)
3. Compute final score S(p) = V(p) + 100 * P(p)
4. Select argmax S(p).
5. Tie-breaker: select shorter prompt_text.

V(p) scoring:

* +40 if R1 says X is single most natural one-word completion
* +30 if no close alternative within top3
* +20 if prompt enforces one-word answer
* +10 if fluent
* If any hard fail: V(p)=0

### 3.7 Prompt validation (mandatory DeepSeek R1)

For each candidate prompt p with target X, call DeepSeek R1 to return strict JSON:

* is_one_word_answer_enforced
* best_one_word_answer
* top3_one_word_answers
* is_X_best
* ambiguity_score (0–10)
* leaks_answer
* naturalness_score (0–10)
* comments

Acceptance rules:

* is_X_best == true
* ambiguity_score <= 3
* leaks_answer == false
* is_one_word_answer_enforced == true
* naturalness_score >= 7

Manual spot checks:

* sample 50 accepted prompts per category
* if >10% flagged -> tighten filters and regenerate

### 3.8 Pressure gating and balancing

After acceptance, compute baseline pressure P0 (Section 7 below).

* Keep only prompts with P0 >= τ.
* Start τ=0.20 and adjust until 500 prompts per category.
* Balance per category across 5 bins of P0: [0–0.2), [0.2–0.4), [0.4–0.6), [0.6–0.8), [0.8–1.0]. Aim 100 per bin; if impossible, maximize coverage and log deviations.

### 3.9 Repetition control across categories

* Do NOT create a global 2,500-word list.
* Enforce cap: same target_word_normalized can appear in at most 2 categories.
* If cap violated, discard candidate and resample.

---

## 4. DeepSeek API Usage Specification

### 4.1 Models and endpoints

* DeepSeek V3.2: used for generating prompts and candidates.
* DeepSeek R1: used for validation scoring.

### 4.2 Request format

* All requests must be logged.
* Store response JSON raw.
* Implement retry with exponential backoff.
* If malformed JSON returned, re-ask with "Return ONLY valid JSON".

### 4.3 Rate limiting

* Implement a token bucket limiter or fixed sleep.
* Save intermediate results so notebook can resume.

---

## 5. Core Correctness Module: Detection + Token Span Mapping (Prefix Incremental Decode Only)

This module must be implemented first and tested heavily.

### 5.1 Requirement summary

For each completion:

1. Determine whether the prohibited word X appears (word-level detection).
2. If yes, determine the exact generated token span(s) responsible.
3. Mapping must be correct even when:

* capitalization differs
* word is tokenized into multiple subword tokens
* leading/trailing whitespace exists
* trailing punctuation exists (e.g., "space.")

### 5.2 Normalization for word-level detection

Define function normalize_for_word_match(s):

1. Unicode NFKC
2. lowercase
3. replace punctuation with spaces including `.,!?;:"'()[]{}<>/\\|-–—_`
4. collapse whitespace to single spaces
5. strip

Word-level detection:

* compute X_norm = normalize_for_word_match(X)
* compute completion_norm = normalize_for_word_match(completion_text)
* split completion_norm on spaces
* word_present = any(token == X_norm for token in tokens)

### 5.3 Prefix incremental decode for token char spans (Method B only)

Given generated token ids ids = [id1, id2, ... idN], compute char spans:

* For i=1..N:

  * s_i = tokenizer.decode(ids[:i], clean_up_tokenization_spaces=False, skip_special_tokens=True)
  * end_i = len(s_i)
* token i span = (end_{i-1}, end_i) with end_0=0
* full decoded string s_full = s_N

This is the only method used.

### 5.4 Find word occurrences in decoded string and map to tokens

1. Build regex for word boundary match in original decoded string s_full:

   * pattern: (?i)(?<![A-Za-z])X(?![A-Za-z]) with X escaped.
2. For each match char interval [a,b):

   * find minimal contiguous token range [t_start,t_end] whose union of spans covers [a,b)
   * decode ids[t_start:t_end+1] and verify it contains the substring corresponding to match
   * verify boundaries are non-letter
3. If verification fails:

   * expand window by ±1 token and retry
   * if still fails, brute force search over all contiguous spans up to length L=8:

     * decode span and check boundary match
4. If word_present==true but mapping fails after brute force:

   * mapping_error=true
   * dump full record to errors/mapping_errors.jsonl
   * increment counter

### 5.5 Mapping must handle trailing punctuation

Because match uses word boundaries against letters only, "space." matches.
Token mapping will cover punctuation token if tokenizer attaches it; verification must allow the punctuation token to exist outside [a,b).

### 5.6 Unit tests (must run before experiment)

Construct synthetic completions and ensure:

* space vs spacetime
* "space." and "space," detected
* " Space" detected
* multi-token tokenization mapping works (create by forcing tokens)
* mapping returns correct token indices

### 5.7 Hard halt condition

After processing any batch of completions:

* if mapping_error_rate > 0.1% OR mapping_error_count > 0 (stricter recommended), halt notebook and require fixing.

---

## 6. Acceptable Surface Forms and Tokenization Set S(X)

This section defines exactly how we treat capitalization/whitespace/punctuation and multi-token splits.

### 6.1 Surface variants list

Generate strings for tokenization enumeration:

* Case variants: X, X.lower(), X.title(), X.upper()
* Whitespace variants: prepend and append spaces:

  * "{v}", " {v}", "{v} ", " {v} "
* Punctuation variants (at minimum trailing period):

  * "{v}.", " {v}.", "{v}. ", " {v}. "
  * (optionally also comma, exclamation, question mark; if included, log them)

### 6.2 Enumerate token sequences

For each variant string s_var:

* encode with tokenizer.encode(s_var, add_special_tokens=False)
* decode back to string and apply normalize_for_word_match
* keep only sequences whose normalized decode equals X_norm
* deduplicate identical token sequences

The resulting set is S(X) = {seq1, seq2, ...}.

### 6.3 Multi-token splits like "spa"+"ce"

No special casing. If tokenizer splits X into multiple tokens, that is naturally included in S(X) when tokenizing the string variant.

---

## 7. Semantic Pressure Computation (Probability Mass for X)

### 7.1 Context definition

For each prompt_text, define context ids for pressure computation as:

* token ids of the full prompt up to and including the substring "Answer:" plus a trailing space if present.
  Ensure prompt format is identical across runs.

### 7.2 Teacher-forced sequence probability

For each token sequence s = [t1..tk] in S(X):

* Compute P(s|context) = Π_i P(ti | context + t1..t{i-1})
  Implementation:

1. Run model(context_ids, use_cache=True) -> logits_1 + past_key_values
2. p(t1) from softmax(logits_1)
3. Append t1; run one-step forward with cache to get logits_2
4. repeat

### 7.3 Union probability

Compute:

* P_sem(X|context) = Σ_{s in S(X)} P(s|context)

### 7.4 Pressure under both conditions

For each prompt:

* P0 = P_sem under baseline prompt_text
* P1 = P_sem under negative-instruction prompt_text

Define suppression metrics:

* Absolute suppression Δ = P0 - P1
* Relative suppression R = (P0 - P1) / max(P0, 1e-9)
* Log suppression L = log(max(P0,1e-12)) - log(max(P1,1e-12))

Store per prompt.

---

## 8. Generation and Behavioral Evaluation

### 8.1 Decoding modes

Two separate decoding regimes are required:

#### 8.1.1 Mechanistic paired runs (deterministic)

* Use greedy/argmax decoding.
* Generate up to max_new_tokens=8.
* Record full internal states for first answer tokens.

#### 8.1.2 Failure probability estimation (stochastic)

* Use nucleus sampling with fixed parameters:

  * temperature=1.0
  * top_p=0.9
* Generate 16 samples per prompt per condition.

Justification: greedy is required to make internal comparisons stable; sampling estimates failure probabilities.

### 8.2 Output constraints and adherence metrics

After generation:

* Determine if output is exactly one word (after normalization, one token in whitespace split)
* If not one word, mark format_nonadherence.

Behavior metrics per prompt (negative instruction):

* violation_rate = fraction of 16 samples where word_present
* format_adherence_rate
* clean_success_rate = fraction where not violated AND format adherent

---

## 9. Mechanistic Measurements

All mechanistic measurements must be computed using greedy decoding (paired runs).

### 9.1 Prompt span tokenization

For each prompt_text, tokenize and record token index spans:

* instr_span: tokens corresponding to the prohibition sentence
* negation_span: tokens corresponding to "Do not use"
* target_mention_span: tokens corresponding to the mention of X inside the instruction
* question_span: tokens for the question

Implement by locating character offsets in prompt string and mapping to token offsets via tokenizer offset mapping.

### 9.2 Attention mass metrics

Capture attention weights A[layer][head][gen_pos][ctx_pos].
For generated position p (typically first token), define:

* Mass(S) = sum_{j in span S} A[...,p,j]

Metrics:

* IAR = Mass(instr_span) / (Mass(instr_span)+Mass(question_span)+1e-9)
* NF  = Mass(negation_span) / (Mass(instr_span)+1e-9)
* TMF = Mass(target_mention_span) / (Mass(instr_span)+1e-9)
* PI  = TMF - NF

Compute:

* global mean across heads/layers
* per layer mean across heads
* per head scores for ranking

### 9.3 Layerwise logit lens for X

For each layer ℓ, obtain hidden state at answer position and compute logits via unembedding.
Compute probability mass assigned to X variants at that layer.
Because X may be multi-token, compute per-step for each token in the chosen X token sequence(s):

* For simplicity, analyze the first generated token’s distribution and compute mass over the set of first tokens of sequences in S(X).
* Additionally, for the actual greedy output sequence, compute token probabilities across layers along the generated path.

Store layer curves for baseline vs negative instruction.

### 9.4 Attention vs FFN contribution decomposition

Hook into transformer blocks to record:

* h_in
* attn_out
* ffn_out
  Compute target probability proxy at:
* h_in
* h_in + attn_out
* h_in + attn_out + ffn_out
  Compute per layer:
* attn_contrib = P(h_in) - P(h_in+attn_out)
* ffn_contrib  = P(h_in+attn_out) - P(h_out)
  Compare success vs failure prompts.

---

## 10. Causal Check: Activation Patching (Only causal method used)

### 10.1 No ambiguity statement

* Only activation patching is performed.
* Head ablation is NOT performed.

### 10.2 Do we rerun all 2,500 prompts?

* No. Activation patching requires forward passes only and is run on a **subset**.
* The subset is selected after the main run based on pressure bins and success/failure outcomes.

### 10.3 Subset selection

Select prompts stratified by:

* Pressure bin (based on P0): 5 bins
* Outcome under negative instruction (using greedy run): success vs failure

Target subset size:

* 50 prompts per bin = 250 prompts total
* Ensure at least 30 failures across subset; if failures are rare in some bins, oversample bins with failures.

### 10.4 Patching procedure

For each selected prompt:

1. Run baseline greedy forward pass and cache:

   * residual stream per layer at answer position
   * attention outputs per layer
   * attention key/value tensors per layer (if accessible)
2. Run negative-instruction greedy forward pass and cache same.

Patch types (run each separately):

* Patch A: Replace neg-instruction residual at layer ℓ with baseline residual (or vice versa) at answer position.
* Patch B: Replace neg-instruction attention output at layer ℓ with baseline attention output.
* Patch C: Patch only instruction segment influence by patching key/value tensors corresponding to instruction token positions.
* Patch D: Patch only question segment influence similarly.

For each patch:

* Recompute next-token distribution at answer position.
* Compute patched P_sem mass for X variants.
* Compute greedy next token and full one-word answer.

Effect metrics:

* ΔP_patch = P_sem_patched - P_sem_original_neg
* Flip indicator: whether greedy answer changes from violating to non-violating or vice versa

Aggregate patch results across layers and bins.

---

## 11. Bootstrap (Prompt-level Resampling)

### 11.1 Why bootstrap is required

Prompts are the experimental units. Different prompts induce different pressures; results must include uncertainty due to prompt selection. Token samples within a prompt are correlated.

### 11.2 What is bootstrapped

Bootstrap across prompts (with replacement) to compute 95% confidence intervals for:

* violation rate per pressure bin
* mean suppression metrics per bin
* mean attention metrics per bin
* patching effect sizes per bin

### 11.3 Bootstrap procedure

For each bootstrap iteration (B=1000):

* sample prompts with replacement within each category or globally (choose one and use consistently)
* recompute metrics
* store
  Compute percentile intervals.

---

## 12. Figures and Tables (Exact)

Generate exactly these plots:

1. Violation rate vs P0 (pressure bins) with CI
2. Suppression (Relative) vs P0 with CI
3. Attention routing metrics (IAR, NF, TMF) vs P0, success vs failure
4. Priming Index PI vs violation probability
5. Layerwise logit lens curves baseline vs negative instruction (success vs failure)
6. Attn vs FFN contributions by layer (success vs failure)
7. Activation patching effects by layer and bin

Export qualitative examples:

* 20 curated failures with:

  * prompt
  * X
  * output
  * token span mapping
  * attention metrics summary

---

## 13. Output File Formats and Folder Layout

All outputs saved under one root folder, e.g. `outputs/experiment_run_YYYYMMDD_HHMM/`.

Required files:

* data/prompts.csv (prompt_id is string key: '{category}_{target_word_normalized}')
* data/prompts_metadata.json
* data/deepseek_validation.jsonl
* runs/completions_samples.jsonl
* runs/completions_greedy.jsonl
* runs/detection_mapping.jsonl
* runs/psem.csv
* runs/pressure_bins.csv
* runs/attention_metrics.csv
* runs/logit_lens.csv
* runs/ffn_attn_decomp.csv
* runs/patching_results.csv
* runs/bootstrap_results.csv
* figures/*.png
* appendix_examples/*.json
* errors/mapping_errors.jsonl

### data/prompts.csv schema

Columns (prompt_id is a string key, not numeric):

* prompt_id (string key: '{category}_{target_word_normalized}')
* question_text (raw cloze question without instruction)
* prompt_text (full baseline prompt with instruction)
* category (idioms|facts|common_sense|creative|ood)
* target_word
* target_word_normalized
* prompt_style_id
* source_trace
* validation_json_ref (e.g. 'facts_validated.jsonl#prompt_id=facts_paris')
* p0 (float; baseline P_sem)
* p1 (float; negative P_sem)
* p0_bin (string like '0.2-0.4')
* v_score (int; validation score)
* p_sem (float; alias of p0 for backward compatibility)
* s_score (float; combined selection score)

All CSVs must have explicit schemas documented in notebook comments.

---

## 14. Paper Writing Specification (Sections to Replace/Create)

The paper must include the rewritten sections:

* Task Setup
* Dataset Construction
* Violation Detection and Token Mapping
* Semantic Pressure
* Mechanistic Measurements
* Causal Intervention (Activation patching)

Use the exact section drafts from the plan previously produced, updated to reflect:

* removal of FORM and association categories
* deterministic selection rule S(p)
* greedy for mechanistic analysis
* activation patching as sole causal method

The agent must embed:

* dataset provenance
* decoding hyperparameters
* all metric definitions
* figure captions aligned to the generated plots

---

## 15. Implementation Notes (Edge catches)

* Always enforce that the prompt does not contain X.
* Always enforce one-word answer instruction.
* Always log failures to parse DeepSeek JSON.
* Always validate that token span mapping works; halt on error.
* Ensure punctuation variants like "X." are included in S(X).
* Ensure whitespace before and after variants are included.

---

## 16. Notebook Execution Checklist

1. Run unit tests for detection/mapping.
2. Build target pool.
3. Generate candidates per category.
4. Validate with DeepSeek R1.
5. Optional Qwen baseline sanity check.
6. Compute P0 and gate/balance.
7. Run greedy paired runs + store internals.
8. Run 16 stochastic samples + store.
9. Run detection/mapping on all completions.
10. Compute behavioral metrics.
11. Compute mechanistic metrics.
12. Run activation patching subset.
13. Run bootstrap.
14. Generate all plots/tables.
15. Export paper-ready artifacts.

---