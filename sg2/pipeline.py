"""The whole v2 job as one resumable chain, shared by sg2/modal_app.py and sg2/setup_pod.sh:

    probe_p0 -> select -> checks -> smoke (64 prompts) -> full run with patching -> analyze
    -> checks against the run's own patching output

    python -m sg2.pipeline --run-dir runs/v2 --candidates data/accepted_candidates.jsonl
    python -m sg2.pipeline --run-dir runs/v2 --prompts data/prompts_v2.jsonl   # skip probe+select

Each finished step leaves a marker (<run-dir>/.done_<step>) and is skipped on rerun; probe_p0
and run_experiment also resume inside a step. The input file is copied into the run folder,
and a rerun with a different input file is refused. Stops at the first failed step (exit 1).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REPO = Path(__file__).resolve().parent.parent


def _gpu_mem_gb() -> float | None:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total",
                                       "--format=csv,noheader,nounits"], text=True)
        return float(out.split()[0]) / 1024
    except Exception:  # noqa: BLE001
        return None


def _adopt(src: Path, dst: Path) -> None:
    """Copy the input into the run folder; refuse if a different one is already there."""
    if dst.exists():
        if dst.read_bytes() != src.read_bytes():
            raise SystemExit(f"STOPPED: {dst} exists and differs from {src}. Use a new run folder.")
        return
    shutil.copyfile(src, dst)


def run_chain(run_dir, model=MODEL, candidates=None, prompts=None, per_cat=None,
              n_brute=3000, after_step=None) -> bool:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    gb = _gpu_mem_gb()
    batch = ["--gen-batch", "256", "--score-batch", "128"] if gb is not None and gb < 40 else []
    print(f"run folder {run}; model {model}; GPU memory "
          f"{'n/a' if gb is None else f'{gb:.0f} GB'}; extra batch args {batch}", flush=True)

    steps = []
    if prompts:
        prompt_file = run / "prompts_v2.jsonl"
        _adopt(Path(prompts), prompt_file)
    else:
        cand = run / "candidates.jsonl"
        _adopt(Path(candidates), cand)
        prompt_file = run / "prompts_v2.jsonl"
        sel = ["-m", "sg2data.select_final", "select", "--probe", str(run / "probe_p0.jsonl"),
               "--candidates", str(cand), "--out-dir", str(run)]
        steps += [
            ("probe", ["-m", "sg2.probe_p0", "--candidates", str(cand), "--out",
                       str(run / "probe_p0.jsonl"), "--model", model]),
            ("select", sel + (["--per-cat", str(per_cat)] if per_cat else [])),
        ]
    steps += [
        ("checks", ["-m", "sg2.checks", "--model", model, "--out", str(run / "checks"),
                    "--n-brute", str(n_brute)]),
        ("smoke", ["-m", "sg2.run_experiment", "--prompts", str(prompt_file), "--out",
                   str(run / "smoke"), "--model", model, "--limit", "64", "--skip-patching", *batch]),
        ("full", ["-m", "sg2.run_experiment", "--prompts", str(prompt_file), "--out", str(run),
                  "--model", model, *batch]),
        ("analyze", ["-m", "sg2.analyze", "--run", str(run)]),
        ("checks_in_run", ["-m", "sg2.checks", "--model", model, "--out", str(run / "checks_after"),
                           "--run-dir", str(run), "--n-brute", "500"]),
    ]
    for name, args in steps:
        marker = run / f".done_{name}"
        if marker.exists():
            print(f"== {name}: already done, skipping", flush=True)
            continue
        print(f"== {name}: started {time.strftime('%F %T')}: python {' '.join(args)}", flush=True)
        t = time.time()
        rc = subprocess.run([sys.executable, "-u", *args], cwd=REPO).returncode
        if rc != 0:
            print(f"STOPPED: step '{name}' failed (exit {rc}). Fix it and rerun the same command; "
                  "finished steps will be skipped.", flush=True)
            if after_step:
                after_step()
            return False
        marker.write_text(time.strftime("%F %T"))
        if after_step:
            after_step()
        print(f"== {name}: finished in {(time.time() - t) / 60:.1f} min", flush=True)
    print(f"ALL DONE {time.strftime('%F %T')}. Results: {run}/analysis/tables.json and figures.",
          flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--candidates", help="accepted candidates; runs probe_p0 + select first")
    g.add_argument("--prompts", help="final prompts; skips probe_p0 + select")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--per-cat", type=int, default=None, help="override 500 per category (tests)")
    ap.add_argument("--n-brute", type=int, default=3000)
    a = ap.parse_args()
    ok = run_chain(a.run_dir, a.model, a.candidates, a.prompts, a.per_cat, a.n_brute)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
