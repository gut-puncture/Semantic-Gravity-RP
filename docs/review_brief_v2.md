# Code review brief: Semantic Gravity Wells v2 GPU job

You are reviewing the code for a rerun of a published interpretability experiment. Find true
bugs only. Do not edit any file; this is a read-only review. Your report goes to another
engineer who will make the fixes under the constraints below, so state each bug so they can
act on it without asking you anything.

## Background

v1 of the paper (arXiv:2601.08070, "Semantic Gravity Wells: Why Negative Constraints
Backfire", LaTeX at `research_paper/arxiv_submission/main.tex`) studied Qwen2.5-7B-Instruct
told "Do not use the word X" for questions whose natural one-word answer is X. An audit found
real bugs in v1 (list in `DESIGN_v2.md`, section "Known v1 bugs this design removes", and in
`/Users/shaileshrana/Documents/Claude/2026-11-24 Reply to Rafal Wrona/paper_audit.md`):
- the priming taxonomy had no base rate among successes, and its rule differed from the paper;
- activation patching reused a mutated KV cache across patch runs;
- the logit lens applied the final norm twice at the last layer;
- P0 summed overlapping prefix variants and then clipped to 1;
- confidence intervals resampled samples instead of prompts; R^2 was undefined;
- no chat template; batched sampling without left padding;
- figure numbers typed by hand; data not published.

v2 reruns everything with fixed code. `DESIGN_v2.md` is the single source of truth for the
method. `sg2/README.md` (section "Decisions") records where the code had to settle details
the design left open; treat those as intended unless one is actually wrong.

## What to review

- `sg2/core.py`: model loading, prompt encoding (chat template), P0/P1 scoring, generation,
  violation detection, logit lens, attention spans, DLA, patching.
- `sg2/run_experiment.py`: the full measurement run (resumable, writes Parquet).
- `sg2/analyze.py`: statistics, tables.json and figures.
- `sg2/checks.py`: golden checks (lens at last layer equals real output; P0 equals a
  brute-force estimate; patching nothing is exact; spans decode correctly; padding consistent;
  DLA sums correctly).
- `sg2/probe_p0.py` and `sg2data/select_final.py`: baseline-P0 probe on 3,846 candidates and
  the balanced selection of 2,500 prompts (500 per category, 100 per P0 bin, nearest-bin fill,
  seed 42).
- `sg2/pipeline.py`: the chain probe -> select -> checks -> smoke (64 prompts) -> full ->
  analyze -> checks on the run's own output. Resumable via `.done_<step>` markers.
- `sg2/setup_pod.sh`: installs pinned packages in a uv venv, downloads the model, starts the
  chain in tmux, logs to `<run>.log`.
- `sg2/colab_job.py`: runs on the Colab VM as one kernel cell. Unpacks the code, runs
  setup_pod.sh, watches the chain, copies the run folder to Google Drive every 5 minutes and
  at the end, then releases the GPU runtime itself (`google.colab.runtime.unassign()`). Guards:
  chain failure (20-minute grace, `/content/KEEP` keeps the VM), 40 minutes with no log
  output, 5-hour total limit.
- Ignore `sg2/modal_app.py` (not used) and everything under `src/`, `notebooks/`, `outputs/`
  (v1 code and old test outputs).

The whole chain already ran end to end on a Mac (MPS/CPU) with Qwen2.5-0.5B-Instruct on a
handful of prompts, and all golden checks passed there. It has never run on CUDA or on the
7B model.

## Where it will run

One Colab GPU, whichever is free first: L4 (24 GB) most likely, possibly A100 (40 or 80 GB)
or G4 (RTX Pro 6000, 96 GB). bf16 weights (~15 GB), eager attention (needed for attention
weights). Python 3.12, torch 2.15.0, transformers 5.24.1 (pinned in setup_pod.sh). The
machine must finish and save results by about 13:45 IST today; the owner's laptop goes off
at 14:00 and the job must not need it after it starts. Paid compute is scarce, so GPU time
matters.

## What counts as a true bug (report these)

1. Anything that makes a reported number wrong or different from DESIGN_v2.md: wrong
   probabilities, wrong token positions, padding leaking into read positions, wrong spans,
   double-applied norms, wrong bootstrap unit, wrong bin edges, selection that uses P0 in a
   way the design forbids, outcome labels mixed up, off-by-one layers, the v1 bugs above
   surviving in any form.
2. Anything that would crash or hang the run on CUDA/7B, or on an L4's 24 GB (out of memory
   at the batch sizes used; `pipeline.py` lowers batches when GPU memory < 40 GB), or with
   multi-token targets, targets with unusual characters, empty generations, or bins with too
   few prompts.
3. Anything that loses or fails to save results: outputs not written, written after the GPU
   is released, not copied to Drive, resume that silently skips or duplicates work, the
   colab_job guard releasing the GPU too early or never.
4. Clear GPU waste that changes run time materially: the GPU idle while the CPU works, many
   small forward passes where one batched pass computes the same thing, recomputing what is
   already computed, batch sizes far below what fits. Only where the fix leaves every
   computed number unchanged.
5. Golden checks that cannot catch the mistake they are meant to catch.

## What not to report

Style, naming, comments, refactors, type hints, missing unit tests, hypothetical issues that
need inputs this job never sees, and method choices that are deliberate in DESIGN_v2.md or
the README "Decisions" list (unless one is actually wrong, in which case say why).

## Report format

A numbered list, most severe first. For each bug: file:line; what is wrong; a concrete
scenario showing the wrong output, crash or loss; severity (results-wrong / crash / data-loss
/ GPU-waste / weak-check); and the smallest fix that keeps the method unchanged. Then one
short paragraph on anything you could not verify. If you find no true bugs in an area, say
so in one line. Keep it under about 1,500 words.
