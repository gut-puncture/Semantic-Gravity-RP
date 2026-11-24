"""Run the whole v2 job on Modal: checks (bf16) -> 64-prompt smoke test -> full run with
patching -> analysis. All outputs go to the Modal volume `sg2-runs` under <run-name>/.

    modal run --detach sg2/modal_app.py --prompts data/prompts_v2.jsonl --run-name v2
    modal volume get sg2-runs v2 runs/v2

--detach keeps the job running if your laptop sleeps or disconnects (without it, closing the
terminal stops the app). Resumable: run the same command again; finished steps are skipped
(marker files) and the experiment skips prompts already written. The volume is committed
every 2 minutes and after each step, so a lost container costs at most a few minutes.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
GPUS = ["H100", "A100-80GB", "L40S", "A10"]  # first available wins
HERE = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.15.0", "transformers==5.24.1", "huggingface_hub==2.4.0", "tokenizers==0.23.3",
        "safetensors==0.8.0", "numpy==2.5.5", "pandas==3.1.1", "pyarrow==26.0.1",
        "scipy==1.18.1", "matplotlib==3.11.2",
    )
    .env({"HF_HOME": "/hf", "PYTHONPATH": "/root", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(HERE, "/root/sg2", ignore=["__pycache__", "*.pyc"])
)

app = modal.App("sg2-semantic-gravity", image=image)
hf_vol = modal.Volume.from_name("sg2-hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("sg2-runs", create_if_missing=True)


def _step(name: str, cmd: list[str], run_dir: Path) -> bool:
    """Run one step as a subprocess (output streams to the Modal log); skip it if its
    marker exists. Returns False on failure (no exception, so Modal does not retry a
    deterministic failure such as a failed check)."""
    marker = run_dir / f".done_{name}"
    if marker.exists():
        print(f"== {name}: already done, skipping", flush=True)
        return True
    print(f"== {name}: {' '.join(cmd)}", flush=True)
    t = time.time()
    rc = subprocess.run([sys.executable, "-u", *cmd], cwd="/root").returncode
    runs_vol.commit()
    if rc != 0:
        print(f"STOPPED: step '{name}' failed (exit {rc}). Outputs so far are in volume "
              f"sg2-runs/{run_dir.name}. Fix and rerun the same command.", flush=True)
        return False
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    runs_vol.commit()
    print(f"== {name}: done in {(time.time() - t) / 60:.1f} min", flush=True)
    return True


@app.function(gpu=GPUS, timeout=3 * 60 * 60, volumes={"/hf": hf_vol, "/runs": runs_vol},
              retries=modal.Retries(max_retries=2, initial_delay=10.0), cpu=8, memory=65536)
def run(prompts_text: str, run_name: str) -> None:
    import torch
    from huggingface_hub import snapshot_download

    run_dir = Path("/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts = run_dir / "prompts.jsonl"
    if prompts.exists() and prompts.read_text() != prompts_text:
        raise SystemExit(f"STOPPED: {run_name} already has a different prompts file. "
                         "Use a new --run-name.")
    prompts.write_text(prompts_text)

    subprocess.run(["nvidia-smi"])
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU {torch.cuda.get_device_name(0)}, {gb:.0f} GB", flush=True)
    batch = [] if gb >= 40 else ["--gen-batch", "256", "--score-batch", "128"]

    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    hf_vol.commit()

    stop = threading.Event()

    def committer():  # keep the volume close to the container's state
        while not stop.wait(120):
            try:
                runs_vol.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[volume commit failed: {e}]", flush=True)

    threading.Thread(target=committer, daemon=True).start()
    rd = str(run_dir)
    try:
        steps = [
            ("checks", ["-m", "sg2.checks", "--model", MODEL, "--out", f"{rd}/checks"]),
            ("smoke", ["-m", "sg2.run_experiment", "--prompts", str(prompts), "--out",
                       f"{rd}/smoke", "--model", MODEL, "--limit", "64", "--skip-patching", *batch]),
            ("full", ["-m", "sg2.run_experiment", "--prompts", str(prompts), "--out", rd,
                      "--model", MODEL, *batch]),
            ("analyze", ["-m", "sg2.analyze", "--run", rd]),
            ("checks_in_run", ["-m", "sg2.checks", "--model", MODEL, "--out", f"{rd}/checks_after",
                               "--run-dir", rd, "--n-brute", "500"]),
        ]
        for name, cmd in steps:
            if not _step(name, cmd, run_dir):
                return
    finally:
        stop.set()
        runs_vol.commit()
    print(f"ALL DONE. Download with: modal volume get sg2-runs {run_name} runs/{run_name}")


@app.local_entrypoint()
def main(prompts: str, run_name: str = "v2"):
    run.remote(Path(prompts).read_text(), run_name)
