This plan is structured logically for a coding agent. It breaks the project into **6 Modules** that should be implemented in order. Each module produces artifacts (files/data) required by the next module.

---

# Module 1: Environment, Configuration, and Utilities
**Goal:** Establish the file structure, global constants, and the core shared logic (normalization and model loading) that all subsequent scripts will import.

### 1.1 Global Configuration (`config.py`)
* **Implementation Details:**
    * Define a `CONFIG` dictionary.
    * **Paths:** Define root paths for Google Drive (`/content/drive/...`). Create a function `setup_directories()` that essentially does `os.makedirs(..., exist_ok=True)` for:
        * `data/raw`, `data/candidates`, `data/validated`
        * `runs/mechanistic_trace`, `runs/failure_samples`
        * `assets`
    * **Seeds:** Define `SEEDS = {'python': 42, 'numpy': 42, 'torch': 42}`.
    * **Model Config:** Store model ID (`Qwen/Qwen2.5-7B-Instruct`).
* **Edge Cases:** Check if Drive is mounted; if not, warn or attempt mount.

### 1.2 Shared Utilities (`utils.py`)
* **Normalization Logic:**
    * Implement `normalize_for_match(text)`:
        * Steps: `NFKC` -> `.lower()` -> `replace(punctuation, ' ')` -> `' '.join(split())`.
        * *Note:* Explicitly import `string.punctuation`.
* **Deterministic Seeding:**
    * Create `set_seed(seed)` function that sets `random`, `np.random`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`.
* **Model Loader (Singleton Pattern):**
    * Create a `ModelWrapper` class.
    * Logic: Load model and tokenizer only once. If called again, return existing instance.
    * *Crucial:* Use `device_map="auto"` and `torch_dtype=torch.bfloat16`.
* **Implementation Note:** Ensure `transformers` version check is included (`>=4.37`).

---

# Module 2: Dataset Construction Pipeline
**Goal:** Generate the 2,500 prompts. This is complex, so separate the "mining" from the "validation."

**Candidate pool rule:** For idioms/facts/common_sense, generate **1,000 candidates** (2× the final size). For creative/ood, generate **1,800 candidates** (600 targets × K=3). If a primary source yields fewer than its target count, fill the gap with GPT-5.2 batch fallback using the category-specific schema (model `gpt-5.2-2025-12-11`, text.format `json_object`, reasoning.effort `none`, reasoning.summary `auto`, text.verbosity `low`, store `true`).
After all candidate pools are assembled, validate **all candidates in a single GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch**. Selection is single-pass and non-iterative: if any category ends with fewer than 500 after gating/bin balancing/repetition caps, fill from the remaining highest S(p) prompts and log the shortfall (no regeneration).

**Process streamlining (non-functional refactor):** consolidate the manual steps for candidate collection, gap computation, batch submission, and batch ingestion into a single automated workflow. Do **not** change any filtering, validation, or selection logic; this refactor is purely to remove manual effort and preserve full audit logging for transparency.

### 2.1 API Clients (`api_clients.py`)
* **GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) Batch Client (OpenAI):**
    * Implement a class `OpenAIClient` with batch helpers (`build_batch_request`, `write_batch_requests`, `create_batch`, `wait_for_batch`, `parse_batch_output`).
    * **Logic:** Use batch-only submissions. No synchronous generation calls in the dataset pipeline.
    * **Logging:** All requests/responses must be logged to disk; fail fast if log path is not writable.
    * *Edge Case:* Handle cases where the API returns text instead of the requested JSON or empty output text.
    * Use text.format `json_object` and include the word "json" plus an example in prompts.
    * Use model `gpt-5.2-2025-12-11` with `reasoning.effort="none"`, `reasoning.summary="auto"`, `text.verbosity="low"`, `store=true`.
* **Wikidata Client:**
    * Implement helper functions using `requests`.
    * Include the exact SPARQL queries for Capitals and Currencies defined in the spec.
* **ConceptNet Client (optional; expected unused this run):**
    * Helper to fetch `/c/en/{word}` and parse edges if ConceptNet becomes available.

### 2.2 Category Generators (`data_mining.py`)
* **Logic:** Create a class or function for each category (A, B, C, D, E).
* **Category A (Idioms):** Use existing `data/raw/idioms.csv` (cache), parse, apply regex filter (`len>=3`, `last_word` is alpha).
    * If idiom pool is short, use GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch fallback (return `idiom_full`, `target_word`) and format with I1–I3 templates.
* **Category B (Facts):** Run SPARQL, filter for single-word objects.
    * If facts pool is short, use GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch fallback (return `subject`, `relation`, `target_word`) and format with F1–F3 templates.
* **Category C (Common Sense):** ConceptNet is expected to be down; generate the **entire** pool via GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) batch fallback:
      - Return structured JSON with subject, relation (UsedFor/MadeOf/HasProperty), target_word
      - Build prompts using C1-C3 templates
      - Enforce diversity (no duplicate subject/target, balance relation types)
      - Target 1,000 candidates (prompts_per_category * 2) before validation
* **Category D & E (Generative):**
    * Use `wordfreq` to pick targets (Zipf 3.5-6.0).
    * Use 600 targets and call GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) in batch to get K=3 candidates per target (1,800 candidates).
    * *Crucial:* Do not selection/validate yet. Just save "Candidates".
    * For Creative, seed with `data/raw/writingprompts.txt` (local file); if unavailable, continue with GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) fallback (microstory + target_word) without scenario seeds.
    * For creative/OOD fallback, enforce the same Zipf band on target_word.
    * If OOD generation yields fewer than 1,800 candidates after per-target selection, fill the gap with fallback schema (context/style/target_word).

### 2.3 Validation & Selection (`validator.py`)
* **Logic:** The "Filter Funnel."
* **Deterministic filters (before validation, mandatory):**
    * Reject if target appears in prompt by whole-word match (case-insensitive).
    * Reject if target appears after stripping non-letters from prompt.
    * Reject if creative is not exactly two sentences.
    * Reject if out-of-distribution is not one or two sentences.
    * Reject if target is not strictly alphabetic.
* **GPT-5.2 (`gpt-5.2-2025-12-11`, reasoning.effort `none`) validator (batch-only):**
    * Send all candidate prompts in a single batch. Parse strict JSON response.
    * Calculate $V(p)$ score based on the rubric (+40 if target is best, +20 if one-word format, etc.).
* **Pressure Probe ($P_{sem}$):**
    * Load Qwen. Run the *baseline* prompt (no constraint).
    * Calculate probability of target word (see Module 5 for math).
* **Selection Rule:**
    * For common sense fallback, enforce relation balance and deduplicate by (question_text, target_word) before selection.
    * For every category, apply P_sem gating + 5-bin balancing and then keep the top 500 by S score within bins (diversity via bin balance + repetition caps).
    * If bins are short, fill to 500 with next-highest S(p) prompts and log shortfalls (no regeneration).
* **Global Gating:**
    * Discard if $P_{sem} < 0.2$ (Model doesn't know the answer).
    * Discard if target word has already appeared twice in the dataset (Global uniqueness check).
    * If any category drops below 500 after repetition filtering, proceed with remaining prompts and log the shortfall (no regeneration).
    * If any category drops below 500 after P_sem gating/bin balancing, fill with next-highest S(p) prompts and log bin shortfalls (no regeneration).
* **Output:** Save final list to `data/validated/prompts.jsonl`.

---

# Module 3: Experiment Runner
**Goal:** Execute the model on the validated prompts and save artifacts.

### 3.1 Prompt Formatting (`prompt_builder.py`)
* **Logic:** Simple string templating.
* **Templates:**
    * **Baseline:** "Answer with exactly one English word.\nQuestion: {q}\nAnswer:"
    * **Negative:** "Answer with exactly one English word.\nDo not use the word "{x}" anywhere in your answer.\nQuestion: {q}\nAnswer:"
* *Implementation Note:* Ensure whitespace and newlines are exact.

### 3.2 The Execution Loop (`runner.py`)
* **Logic:** Iterate through `prompts.jsonl`. For each prompt, run two passes.
* **Pass 1: Mechanistic (Greedy)**
    * `model.generate(..., max_new_tokens=1, do_sample=False, output_hidden_states=True, output_attentions=True)`.
    * *Storage:* Save the tuple `(hidden_states, attentions)` to a `.pt` file in `runs/mechanistic_trace/`. Naming convention: `{prompt_id}_{condition}.pt`.
    * *Warning:* These files are huge. You might need to save only the *last* token's hidden state/attention to save space, but the plan calls for "full internal states".
* **Pass 2: Behavioral (Stochastic)**
    * `model.generate(..., max_new_tokens=10, do_sample=True, num_return_sequences=16)`.
    * *Storage:* Save the text strings to `runs/failure_samples/completions.jsonl`.

---

# Module 4: The Correctness Engine
**Goal:** Detect failures and map them to tokens. This must be a standalone module to allow for unit testing.

### 4.1 Detection Logic (`detector.py`)
* **Word Level:**
    * Use decoded string from token ids as authoritative text for detection.
    * Use case-insensitive matching with non-alphanumeric boundaries
      (as defined by `str.isalnum()`).
* **Token Mapping (The Hard Part):**
    * Implement **Prefix Incremental Decoding**.
    * **Algorithm:**
        1.  Decode `ids[0:i]`. Get string length $L_1$.
        2.  Decode `ids[0:i+1]`. Get string length $L_2$.
        3.  Token $i$ spans characters $[L_1, L_2)$.
        4.  Find character offsets of the target word in the full string using
            case-insensitive match + non-alphanumeric boundary checks.
        5.  Find which token spans intersect with the regex character match.
* **Edge Cases:**
    * Trailing punctuation ("space.").
    * Multi-token splits ("sp", "ace").
* **Unit Tests:** Include a block `if __name__ == "__main__":` that runs synthetic tests (e.g., checking if "Space" matches "spacetime" -> False).
  - Include "space2" (should be False) and "space-time" (should be True).

---

# Module 5: Metrics & Analysis
**Goal:** Compute the math for the paper.

### 5.1 Semantic Pressure Math (`metrics_psem.py`)
* **Logic:** Calculate $P_{sem}(X)$.
* **Tokenization Set $S(X)$:**
    * Generate variants: "X", " X", "X.", " X.". Capitalize them.
    * Tokenize all variants.
* **Teacher Forcing Loop:**
    * For a multi-token variant $[t_1, t_2]$:
        * Get prob of $t_1$ from context.
        * Feed context + $t_1$. Get prob of $t_2$.
        * Multiply.
* **Aggregation:** Sum probabilities of all unique valid sequences.

### 5.2 Attention Metrics (`metrics_attn.py`)
* **Logic:** Analyze the greedy `.pt` files.
* **Spans:** Identify token indices for "Do not use", "{X}", and "Question".
* **Formulas:**
    * $NF = \text{Attn(Negation)} / \text{Attn(Instruction)}$.
    * $PI = \text{Attn(Target)} - \text{Attn(Negation)}$.
* *Implementation Note:* Add a small epsilon ($1e-9$) to denominators.

### 5.3 Activation Patching (`patching.py`)
* **Logic:** This is a *new* inference loop, not just analysis of old files.
* **Subset:** Load top 50 prompts per pressure bin.
* **Intervention:**
    * Run Forward Pass A (Baseline) -> Cache activations.
    * Run Forward Pass B (Negative) -> At Layer L, replace `hidden_states` with Cached A.
    * Measure change in $P_{sem}$.
* *Optimization:* Use hooks (`model.register_forward_hook`) to modify the stream during inference.

---

# Module 6: Visualization & Reporting
**Goal:** Create the figures for the paper.

### 6.1 Plotting (`visualize.py`)
* **Input:** The CSVs generated in Module 5.
* **Figure 1 (Failure Curve):** `seaborn.lineplot`. X=Pressure Bin, Y=Violation Rate.
* **Figure 2 (Logit Lens):** X=Layer, Y=$P_{sem}$. Two lines (Baseline vs Negative).
* **Figure 3 (Patching):** Heatmap or Bar chart. X=Layer, Y=$\Delta P_{sem}$.

### 6.2 Paper Artifacts
* Generate a `tables.json` containing the exact numbers for the Results section (e.g., "Mean violation rate for high pressure: X%").

---

### **Implementation Checklist for the Coding Agent**
1.  **Strict Order:** Do not write Module 3 before Module 2 is 100% verified.
2.  **Save Often:** Every long loop (dataset gen, experiment run) must verify it is saving progress to Drive.
3.  **Memory Management:** In Module 3 and 5, explicitly `del` tensors and run `torch.cuda.empty_cache()` inside loops.
4.  **Logging:** Use the `logging` library, not just `print`, so timestamps are captured.
