"""Run the whole v2 job on Modal in one container: probe_p0 -> select -> checks (bf16) ->
64-prompt smoke test -> full run with patching -> analyze (chain in sg2/pipeline.py). All
outputs go to the Modal volume `sg2-runs` under <run-name>/.

    modal run --detach sg2/modal_app.py --candidates data/accepted_candidates.jsonl --run-name v2
    modal run --detach sg2/modal_app.py --prompts data/prompts_v2.jsonl --run-name v2  # skip probe+select
    modal volume get sg2-runs v2 runs/v2

--detach keeps the job running if your laptop sleeps or disconnects (without it, closing the
terminal stops the app). Resumable: run the same command again; finished steps are skipped
(marker files), and the probe and the experiment skip work already written. The volume is
committed every 2 minutes and after each step, so a lost container costs a few minutes.
"""
from __future__ import annotations

import subprocess
import threading
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
    .add_local_dir(HERE.parent / "sg2data", "/root/sg2data", ignore=["__pycache__", "*.pyc"])
)

app = modal.App("sg2-semantic-gravity", image=image)
hf_vol = modal.Volume.from_name("sg2-hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("sg2-runs", create_if_missing=True)


@app.function(gpu=GPUS, timeout=3 * 60 * 60, volumes={"/hf": hf_vol, "/runs": runs_vol},
              retries=modal.Retries(max_retries=2, initial_delay=10.0), cpu=8, memory=65536)
def run(input_text: str, input_kind: str, run_name: str, per_cat: int | None = None) -> None:
    """input_kind: 'candidates' (probe + select first) or 'prompts' (final prompt set)."""
    from huggingface_hub import snapshot_download

    from sg2 import pipeline

    subprocess.run(["nvidia-smi"])
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    hf_vol.commit()

    src = Path(f"/tmp/{input_kind}.jsonl")
    src.write_text(input_text)
    stop = threading.Event()

    def committer():  # keep the volume close to the container's state
        while not stop.wait(120):
            try:
                runs_vol.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[volume commit failed: {e}]", flush=True)

    threading.Thread(target=committer, daemon=True).start()
    try:
        # A failed step returns False rather than raising, so Modal does not retry a
        # deterministic failure such as a failed check; retries cover lost containers.
        ok = pipeline.run_chain(
            f"/runs/{run_name}", MODEL,
            candidates=src if input_kind == "candidates" else None,
            prompts=src if input_kind == "prompts" else None,
            per_cat=per_cat, after_step=runs_vol.commit)
    finally:
        stop.set()
        runs_vol.commit()
    if ok:
        print(f"Download with: modal volume get sg2-runs {run_name} runs/{run_name}")


@app.local_entrypoint()
def main(candidates: str = "data/accepted_candidates.jsonl", prompts: str = "",
         run_name: str = "v2", per_cat: int = 0):
    path, kind = (prompts, "prompts") if prompts else (candidates, "candidates")
    print(f"Sending {kind} file {path} to run '{run_name}'")
    run.remote(Path(path).read_text(), kind, run_name, per_cat or None)
