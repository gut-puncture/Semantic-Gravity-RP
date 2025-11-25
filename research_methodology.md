# Research Methodology

**Title:** *Semantic Gravity: Quantifying the Efficiency Gap in Negative Constraint Adherence*

### 1. Abstract & Core Thesis
**The Problem:** Current safety benchmarks treat all negative constraints equally (e.g., "Don't say 'kill'" is treated the same as "Don't say 'the'"). This is a flaw in measurement.
**The Theory ("Semantic Gravity"):** We hypothesize that the difficulty of adhering to a negative constraint is functionally dependent on the **contextual probability** of the forbidden token. When a model predicts a token with high confidence (High Logit), the internal "pressure" to generate it overrides the instruction to suppress it.
**The Gap:** While massive reasoning models (GPT-5 class) utilize "System 2" thinking to override this pressure, efficient models (7B-8B class) lack this inhibition mechanism, exhibiting a predictable "Collapse Threshold."

### 2. The Metrics
We will define two key metrics for this study:
1.  **Semantic Pressure ($P_{sem}$):** The probability assigned to a specific token $t$ in context $C$ *before* any negative constraint is applied.
    *   *Formula:* $P_{sem} = \sum P(t_i | C)$ for all tokens $t_i$ that represent the concept (e.g., " Apple", "apple", "Apple").
2.  **Adherence Failure Rate ($R_{fail}$):** The binary state (0 or 1) of whether the model generates token $t$ in the generated output despite the constraint.

### 3. Experiment Design

#### Experiment 1: The Collapse Curve (Mechanism)
*   **Hypothesis:** Failure rate is not linear. It follows a **Sigmoid function**. There exists a specific "Tipping Point" (e.g., $P_{sem} > 0.4$) where efficient models suffer a catastrophic loss of control.
*   **Model:** Qwen-2.5-7B-Instruct (White Box).
*   **Procedure:**
    1.  Input 100+ prompts across the probability spectrum (0.01 to 0.99).
    2.  Condition A (Baseline): Run *without* constraint. Record $P_{sem}$ (The "Pressure").
    3.  Condition B (Constraint): Run *with* "Do not use [Word]."
    4.  **Analysis:** Plot $P_{sem}$ (X-axis) vs. $R_{fail}$ (Y-axis). Fit a Logistic Regression curve to find the exact "Collapse Threshold."

#### Experiment 2: The Efficiency Gap (Validation)
*   **Hypothesis:** "Semantic Gravity" is the defining difference between "Smart" and "Efficient" models.
*   **Models:** GPT-5 Nano vs. GPT-5 Base (Black Box / API).
*   **Procedure:** Run the exact same dataset from Exp 1.
*   **Analysis:** We expect GPT-5 Base to show a flat line (near 0% error). We expect GPT-5 Nano to show the same Sigmoid curve as Qwen.
*   **Conclusion:** This proves that **inhibition is an emergent property of scale**, meaning efficient models are inherently unsafe for High-Gravity tasks unless mitigated.

#### Experiment 3: Anchor Displacement (Mitigation)
*   **Hypothesis:** We can artificially lower $P_{sem}$ (and thus save the model) not by shouting "STOP," but by providing a new Semantic Anchor.
*   **Procedure:** Take the top 50 "Failed" prompts from Exp 1.
    *   *Method A (Control):* "Don't say X."
    *   *Method B (Anchor Displacement):* "Don't say X. Use [Synonym Y] instead."
*   **Analysis:** We measure the reduction in Failure Rate.


