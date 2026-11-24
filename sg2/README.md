# sg2: Semantic Gravity v2 GPU experiment

The code that produces every model measurement in `DESIGN_v2.md` ("Prompts given to Qwen",
"Measurements per prompt", "Analysis"). It is separate from the v1 code in `src/`.

| File | What it does |
|---|---|
| `core.py` | Prompt templates, exact P0/P1, hooked forward pass (logit lens, attention, DLA, patching), batched generation |
| `run_experiment.py` | The one entry point: measurements 1–7 for every prompt, then patching (8) on a subset |
| `analyze.py` | Every number and figure in "Analysis", from the saved outputs only |
| `checks.py` | Golden and integration checks against the real model |
| `test_prompts.jsonl` | 12 hand-written prompts across the five categories |

## One-command runs

- **Modal**: `modal run --detach sg2/modal_app.py --prompts data/prompts_v2.jsonl --run-name v2`,
  then `modal volume get sg2-runs v2 runs/v2`. It uses one GPU from H100, A100-80GB, L40S, A10
  (first available), caches the weights in volume `sg2-hf-cache`, has a 3 h timeout, and
  runs checks → smoke → full run → analysis → in-run checks. `--detach` keeps it running
  when your laptop disconnects. Rerunning the same command resumes. Not yet run on Modal;
  locally only the app graph was built.
- **Plain GPU box** (Prime Intellect pod): `bash sg2/setup_pod.sh data/prompts_v2.jsonl runs/v2`
  installs everything, downloads the weights, and runs the same steps in tmux session `sg2`,
  logging to `runs/v2.log`. It stops at the first failed step, and rerunning it resumes.

## Set up a fresh Linux GPU box by hand

Tested on macOS (Apple M4, MPS) with Python 3.12 and exactly these versions. It has not yet
been run on CUDA, so run the checks and the 64-prompt smoke test first.

```bash
git clone <repo> && cd 29_Semantic_Gravity_v2 && git checkout v2
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 transformers==5.24.1 huggingface_hub==2.4.0 tokenizers==0.23.3 \
    safetensors==0.8.0 numpy==2.5.5 pandas==3.1.1 pyarrow==26.0.1 scipy==1.18.1 matplotlib==3.11.2
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"   # must print True
# If False, the box's NVIDIA driver is older than the wheel's CUDA build: reinstall torch from
# https://download.pytorch.org/whl/cu128 (pip install torch==2.15.0 --index-url <that URL>).
huggingface-cli download Qwen/Qwen2.5-7B-Instruct      # ~15 GB
```

## Run

```bash
export PYTHONPATH=$PWD
# 1. Golden checks on the 7B model (about 5 min). On an H100 add --dtype float32 for the
#    spec's fp32 comparison; on an L4 (24 GB) bf16 is the only option.
python -m sg2.checks --model Qwen/Qwen2.5-7B-Instruct --out runs/checks_7b

# 2. Smoke test: 64 prompts, read the prompts/s line, then delete the folder.
python -m sg2.run_experiment --prompts data/prompts_v2.jsonl --out runs/smoke --limit 64 --skip-patching

# 3. Full run (resumable: rerun the same command after any crash or stop).
python -m sg2.run_experiment --prompts data/prompts_v2.jsonl --out runs/v2 \
    --model Qwen/Qwen2.5-7B-Instruct            # L4: add --gen-batch 256 --score-batch 128

# 4. Analysis (CPU, about a minute) and the in-run patching checks.
python -m sg2.analyze --run runs/v2
python -m sg2.checks --model Qwen/Qwen2.5-7B-Instruct --out runs/checks_7b --run-dir runs/v2
```

The prompts file is JSONL with `prompt_id, category, question, target`. Outputs in `--out`:
`measure/part-*.parquet` (one row per prompt, measurements 1–7), `patch_subset.json`,
`patch/part-*.parquet` (measurement 8), `run_config.json` (invariants plus every start, with
git commit and versions), `run.log`, and `analysis/` (tables.json and fig1–fig6 PNGs).

Resuming skips prompts already written. Each 64-prompt chunk is written atomically, so a crash
loses at most one chunk. A resume refuses to start if the model, dtype, prompt templates,
sampling settings or seed differ from the first start; batch sizes, chunk size and hardware may
change. Batches that run out of GPU memory are halved and retried automatically.

## Measured on the Mac (Qwen2.5-0.5B, MPS, fp32, 12 test prompts)

| Step | Time |
|---|---|
| Stage 1 (measurements 1–7), one 12-prompt chunk | 17.7 s = 0.68 prompts/s (sampling 12.2 s, P0/P1 3.3 s, hooked pass 1.4 s, greedy 0.8 s) |
| Stage 2 (patching, 49 forward rows per prompt) | 0.92 prompts/s |
| `checks.py` | 3 min 10 s, almost all of it the 9,000 brute-force samples |
| `analyze.py` (1,000 bootstrap draws) | 10 s |

## Expected runtime for 2,500 prompts on Qwen2.5-7B

From token counts, with prompts of about 70 tokens (the test prompts are 47–70 with the chat template):

- P0/P1 scoring: 2 conditions × 6 variants × 72 tokens = 864 tokens per prompt
- Hooked pass: 2 × 70 = 140; greedy: 2 × 78 = 156
- Sampling: 2 × 16 × 80 = 2,560 (generate re-runs the prompt for each of the 16 samples)
- Total about 3,700 token-forwards per prompt, times 14 GFLOP per token for 7B, is 52 TFLOP
  per prompt, or 130 PFLOP for 2,500 prompts. Patching adds 250 × 58 rows × 70 tokens × 14
  GFLOP = 14 PFLOP.

At realistic Hugging Face throughput (about 30% of peak bf16, plus Python and decode-step
overhead), that is:

| GPU | Effective | Stage 1 | Patching | Total |
|---|---|---|---|---|
| H100 80 GB | ~300 TFLOP/s | ~8 min | ~1 min | **15–25 min** with loading and overhead |
| L4 24 GB | ~40 TFLOP/s | ~55 min | ~6 min | **1–1.5 h** |

The Mac numbers do not scale to these directly (the 0.5B run is dominated by per-step overhead),
which is why the estimate is FLOP-based; the smoke test's prompts/s line gives the real figure
within a minute. Sampling is about 70% of the FLOPs.

## Golden checks on Qwen2.5-0.5B (all passed, fp32 on MPS)

- (a) Logit lens at the last layer vs the model's real distribution: max abs diff 0 within the
  same pass, 9.5e-6 against a separate unbatched plain forward (24 distributions, tolerance 1e-3).
- (b) Max P over all 24 prompt/condition pairs 0.952 ≤ 1. Brute force, 3,000 samples at
  temperature 1, top_p 1: P0 0.339 vs 0.326 (z −1.4), 0.664 vs 0.645 (z −2.2), 0.222 vs 0.233
  (z +1.4). Counting any tokenisation instead of the canonical one changes the frequency by ≤ 0.001.
- (c) Patching nothing: diff 0 exactly. Last-layer residual patch vs baseline distribution: 0.
  Layer-0 attention patch changes the output (0.0094), so the hook is not a no-op. In the run
  itself, against stage-1 values computed in a different batch: ≤ 3.5e-6 and ≤ 5.1e-6.
- (d) All 12 target-mention, "Do not" and question spans decode to the expected text.
- (e) Greedy output identical alone and inside a left-padded batch for all 24 prompts (lengths 47–70).
- (f) Extra: embedding term plus per-layer attention and MLP DLA sums to the real target logit
  (max relative diff 1.3e-6).

## Decisions

Where the spec was ambiguous or would have been wrong as written:

1. **Clean generation settings.** Qwen's hub defaults (temperature 0.7, top_k 20, top_p 0.8,
   repetition_penalty 1.1) would silently change both greedy and sampled outputs. The model's
   generation config is replaced at load; sampling uses exactly temperature 1.0, top_p 0.9, no
   top-k, no repetition penalty; greedy uses no repetition penalty.
2. **Padding.** Generation uses left padding (asserted). The scoring passes (P0/P1, lens,
   attention, DLA, patching) use right padding and read each row at its own last token, so pad
   tokens can never reach the positions read; check (e) covers the left-padded path.
3. **Target token t\***: the first token of the surface variant with the highest P0 + P1. Using
   the baseline alone fails when the negative prompt shifts case ("rain" → "Rain"). The
   **target set** is the distinct first tokens of all variants, the next-token analogue of
   P0/P1. Analysis uses the set for the logit lens and patching, and t\* for DLA, which needs a
   single direction. For multi-token words the first token is a loose proxy (Qwen splits
   "Jupiter" into "J" + "upiter"). So every mechanistic result (suppression, signatures,
   lens, DLA and attention curves, patching) is reported twice in `tables.json` under
   `mechanistic`: on all prompts and on prompts whose preferred spelling is one token
   (`target_single_token`, stored per prompt). The figures for that subset are in
   `analysis/single_token/`.
4. **Patching measures the next token only**, as the spec requires, so its "P1" is the
   next-token probability of the target set (and of t\*), not the multi-token P1. "Patching
   nothing reproduces P1 exactly" is checked against that unpatched next-token probability.
5. **Prompt-level outcome = greedy failure.** `tables.json` cross-tabulates it against the
   majority of the 16 samples.
6. **"TMF > NF"** compares attention mass averaged over all layers (and heads). The
   target-mention span is the tokens overlapping X's characters inside the quotes. The question
   span may include a final token that merges the closing punctuation with the newline.
7. **DLA** projects onto W_U[t\*] scaled by the final norm's gain and the real final residual's
   RMS (the spec's definition; figures use it). A centred version (W_U[t\*] minus the mean
   unembedding row, which removes the part that shifts every logit equally) is saved too.
8. **Patching subset rule.** Per P0 bin, 25 greedy failures and 25 successes chosen at random
   (seed 42), filling from the other outcome when one is short; saved to `patch_subset.json`,
   or replaced with `--patch-ids`.
9. **Seeds.** Sampling is seeded per chunk from sha256(seed, first prompt id), and the seed is
   stored per row. Rerunning with the same chunking reproduces samples on the same hardware.
   GPU sampling is not bit-identical across GPU types.
10. **Word boundary** means no letter or digit on either side, for both the violation detector
    and the "next token does not start with a letter or digit" rule in P0. "Paris-based" and
    "Paris's" count as the word Paris.
11. **Check tolerances in bf16.** The spec's 1e-3 lens comparison is for fp32. In bf16, the
    comparison against a separately batched forward uses 2e-2, because bf16 kernels differ with
    batch shape; same-pass equality must still be exact. Run checks with `--dtype float32` on
    an H100 for the strict version.
12. **R² of the fit to bin means** uses the five design bins, with the curve evaluated at each
    bin's mean P0. It is labelled separately from the sample-level McFadden R².
13. **Matched suppression comparison**: the within-bin differences are pooled with weights
    n_fail·n_success/(n_fail+n_success), and the CI comes from resampling prompts within bins.
