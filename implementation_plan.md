# Implementation Plan

**Environment:** Google Colab (T4 GPU).
**Files:** `experiment_runner.py` (One monolithic script to prevent import errors).

### Step 1: The "High-Variance" Data Strategy (CRITICAL STEP)
**Potential Bug:** If we generate 100 prompts that are all "The sky is...", the model will have 0.99 probability for all of them. We will have no data for the "Medium Pressure" (0.5) range, and the graph will look empty.

**The Protocol:** We must generate prompts in **5 Distinct Gravity Buckets** to ensure a beautiful X-axis distribution.

*   **Bucket A: Deterministic Idioms (Pressure: 0.9 - 1.0)**
    *   *Instruction:* "Create sentences that are fixed idioms."
    *   *Example:* "A blessing in..." -> Disguise.
*   **Bucket B: Factual Recall (Pressure: 0.6 - 0.9)**
    *   *Instruction:* "Create historical or scientific facts."
    *   *Example:* "The capital of France is..." -> Paris.
*   **Bucket C: Common Sense/Categorical (Pressure: 0.3 - 0.6)**
    *   *Instruction:* "Create questions with 2-3 likely answers."
    *   *Example:* "Name a common pet that isn't a dog." -> Cat (Probability split between Cat/Hamster/Fish).
*   **Bucket D: Creative Writing (Pressure: 0.1 - 0.3)**
    *   *Instruction:* "Create open-ended story prompts."
    *   *Example:* "The alien spaceship landed in the..." -> Park/Desert/City (Low probability for any single word).
*   **Bucket E: Counter-Intuitive / OOD (Pressure: Variable)**
    *   *Instruction:* "Create scenarios that contradict training data."
    *   *Example:* "In a world where grass is blue and the sky is green, the color of the lawn is..." -> Blue.
    *   *Note:* This tests if the model follows the *context* (Blue) or the *semantic memory* (Green).

**Execution:** We will use a helper script to call an API (or you can manually use ChatGPT) to generate 20 examples for each bucket before running the main code.

### Step 2: The "White Box" Analyzer (Qwen)
**Potential Bug (The "Token Split" Error):**
*   *The Bug:* You forbid the word "Apple". The model predicts " Apple" (with a leading space). The code checks `if "Apple" in top_token`: it returns False. The data becomes garbage.
*   *The Fix:* We must sum the probabilities of the **Top 5 tokens**. If *any* of the top 5 tokens contain the target string (stripped of whitespace/capitalization), we add their probabilities together to get the true $P_{sem}$.

**Potential Bug (Memory Leak):**
*   *The Bug:* Loop 100 times. PyTorch accumulates gradients. RAM fills up. Colab crashes at prompt #40.
*   *The Fix:* Inside the loop, explicitly call `torch.cuda.empty_cache()` and `del outputs` after every iteration.

### Step 3: The "Black Box" Runner (GPT-5 Family)
**Potential Bug (The "Chatty Model" Error):**
*   *The Bug:* You ask "Write a story without 'Apple'". GPT-5 says: "Sure! Here is a story that adheres to your constraint: Once upon a time..."
*   *The Issue:* If we just search for "Apple", we might find it in the intro text, or we might miss it if the model hallucinates.
*   *The Fix:* We will separate the prompt into System and User.
    *   *System:* "You are a completion engine. Do not output conversational filler. Just output the completion."
    *   *Check:* We perform a case-insensitive string search (`.lower()`) on the final output.

### Step 4: Analysis & Curve Fitting
**Potential Bug (The "Bad Fit"):**
*   *The Bug:* Real data is noisy. The points are everywhere. The Sigmoid curve fails to converge.
*   *The Fix:* We will use **Binning** before fitting.
    *   Group the 100 data points into 10 bins of probability (0.0-0.1, 0.1-0.2...).
    *   Calculate the average Failure Rate for each bin.
    *   Fit the Sigmoid curve to these 10 averages (points) rather than the 100 raw dots. This guarantees a smooth, professional-looking graph for the paper.

### Step 5: The "Savepoint" Protocol
**Critical Instruction:**
Do not wait until the end to save data.
The script must write to `experiment_results.csv` in **Append Mode (`mode='a'`)** after *every single prompt*.
*   *Why:* If Colab disconnects at 99/100, you will lose everything if you save at the end. With Append Mode, you just restart the script and skip the ones already done.

### Step 6: Experiment 3 (The Mitigation Pass)
**Goal:** Prove that we can manipulate "Semantic Gravity" using "Anchor Displacement."
**Logic:** We take the subset of prompts where Qwen failed in Exp 1 (or the top 50 highest pressure prompts). We run them again with the modified prompt structure.

**The Protocol:**
1.  **Filter:** Select prompts where `Pressure_Score > 0.5`.
2.  **Prompt Modification:**
    *   *Baseline Prompt:* "Complete the sentence. Constraint: Do not use the word '{target}'."
    *   *Mitigation Prompt:* "Complete the sentence. Constraint: Do not use the word '{target}'. Instead, use the word '{synonym}'."
3.  **Measurement:**
    *   We do not just check if it failed. We check the **Logit Drop**.
    *   *Question:* Did the probability of `{target}` decrease in the Mitigation Prompt compared to the Baseline Prompt?

**Potential Bug (The "Hallucinated Synonym" Error):**
*   *The Bug:* If we let the code auto-generate synonyms using NLTK/WordNet on the fly, it might pick a synonym that doesn't make sense in context (e.g., Target: "Apple", Synonym: "Company"). If the synonym doesn't fit the sentence, the model will ignore it, and the experiment fails.
*   *The Fix (Pre-Validation):* We must generate the synonyms **during Step 1 (Data Generation)**. When we ask the Teacher LLM (GPT-4o) to generate the prompts, we explicitly ask: "Return JSON: `{prompt: "...", target: "...", valid_synonym: "..."}`". This ensures the synonym is contextually perfect before we even start the experiment.

**Potential Bug (The "Instruction Conflict"):**
*   *The Bug:* By mentioning the forbidden word *and* the synonym, the prompt becomes longer. The model might just attend to the last word it saw.
*   *The Fix:* We use a strict formatting template:
    `System: You are a precise assistant.`
    `User: {Context}`
    `Instruction: 1. You must NOT use the word '{Target}'. 2. You MUST use the word '{Synonym}' to complete the thought.`
    (Numbering the instructions breaks the semantic flow and forces attention to both).

### Step 7: Final Data Structure & Export
**Goal:** Ensure the CSV is ready for the paper without manual cleaning.

**The Schema:**
Your final `results.csv` must have these exact columns. If you miss one, you can't make the graphs.

| Column Name | Description | Used For |
| :--- | :--- | :--- |
| `prompt_id` | Unique ID (1-100) | Tracking |
| `bucket` | A/B/C/D/E (Gravity Level) | Exp 1 X-Axis Distribution |
| `target_word` | The forbidden word | Validation |
| `baseline_logit` | Logit of target *without* constraint | **Exp 1 X-Axis (Pressure)** |
| `baseline_prob` | Softmax probability (0.0-1.0) | Readability |
| `exp1_failure` | 0 or 1 (Did it fail with simple constraint?) | **Exp 1 Y-Axis** |
| `exp1_logit` | Logit of target *with* simple constraint | Mechanism Analysis |
| `exp3_failure` | 0 or 1 (Did it fail with Anchor?) | **Exp 3 Y-Axis** |
| `exp3_logit` | Logit of target *with* Anchor | **Exp 3 Proof of Mechanism** |

---

### FINAL CHECKLIST: HOW TO EXECUTE IN "ONE SITTING"

1.  **Minute 0-10 (Setup):**
    *   Open Colab -> Runtime -> Change to **T4 GPU**.
    *   Mount Google Drive (to save CSVs).
    *   Paste the **Master Script** (I will provide this next).
    *   Input your OpenAI Key (for the Data Generation helper).

2.  **Minute 10-20 (Data Gen):**
    *   Run the `generate_dataset()` function.
    *   **Human Step:** Open the generated JSON file. Read the first 10 and the last 10. Do they make sense? Is the "Synonym" actually a synonym? If yes, proceed.

3.  **Minute 20-50 (Qwen Analysis - Exp 1 & 3):**
    *   Run the `run_experiment()` function.
    *   It will load Qwen (takes ~2 mins).
    *   It will loop through the 100 prompts.
    *   *Watch the progress bar:* Ensure `failure_rate` isn't 100% or 0%. If it is, stop. Something is wrong with the token checking (e.g., casing issue).

4.  **Minute 50-60 (GPT Comparison - Exp 2):**
    *   Run the separate `run_gpt_benchmark()` script.
    *   This is fast (API calls).

5.  **Minute 60+ (Writing):**
    *   Download `results.csv`.
    *   Run the `plot_results()` function (I will include this in the script).
    *   Take the 3 generated PNGs.
    *   Paste them into your Overleaf template.