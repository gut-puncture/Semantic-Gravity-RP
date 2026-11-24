#!/usr/bin/env bash
# Set up a plain Linux GPU box (e.g. Prime Intellect pod, Ubuntu + CUDA image) and run the
# whole v2 job inside tmux: checks -> 64-prompt smoke test -> full run -> analysis.
#
#   bash sg2/setup_pod.sh data/prompts_v2.jsonl runs/v2
#
# Run from the repo root. Everything is logged to <run-dir>.log. The job runs in tmux
# session "sg2"; detach with Ctrl-b d, reattach with `tmux attach -t sg2`. Rerunning the
# script resumes: finished steps are skipped and the experiment skips finished prompts.
# It stops with a clear message at the first failed step.
set -euo pipefail

PROMPTS="${1:?usage: setup_pod.sh <prompts.jsonl> <run-dir>}"
RUN="${2:?usage: setup_pod.sh <prompts.jsonl> <run-dir>}"
MODEL="${SG2_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO" HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
LOG="$RUN.log"

# ---------------------------------------------------------------- pipeline (runs in tmux)
if [[ "${SG2_PIPELINE:-}" == "1" ]]; then
  source .venv/bin/activate
  mkdir -p "$RUN"
  GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  BATCH=()
  if (( GB < 40000 )); then BATCH=(--gen-batch 256 --score-batch 128); fi
  step() {  # step <name> <python args...>; skipped if its marker exists
    local name="$1"; shift
    if [[ -f "$RUN/.done_$name" ]]; then echo "== $name: already done, skipping"; return; fi
    echo "== $name: started $(date '+%F %T')"
    if ! python -u "$@"; then
      echo "STOPPED: step '$name' failed. Read $LOG, fix, and rerun the same command."
      exit 1
    fi
    date '+%F %T' > "$RUN/.done_$name"
    echo "== $name: finished $(date '+%F %T')"
  }
  nvidia-smi
  step checks -m sg2.checks --model "$MODEL" --out "$RUN/checks"
  step smoke -m sg2.run_experiment --prompts "$PROMPTS" --out "$RUN/smoke" --model "$MODEL" \
       --limit 64 --skip-patching "${BATCH[@]}"
  step full -m sg2.run_experiment --prompts "$PROMPTS" --out "$RUN" --model "$MODEL" "${BATCH[@]}"
  step analyze -m sg2.analyze --run "$RUN"
  step checks_in_run -m sg2.checks --model "$MODEL" --out "$RUN/checks_after" --run-dir "$RUN" --n-brute 500
  echo "ALL DONE $(date '+%F %T'). Results: $RUN/analysis/tables.json and figures."
  exit 0
fi

# ---------------------------------------------------------------- setup (runs once)
[[ -f "$PROMPTS" ]] || { echo "prompts file not found: $PROMPTS"; exit 1; }
SUDO=""; [[ $(id -u) -ne 0 ]] && SUDO="sudo"
if ! command -v tmux >/dev/null; then $SUDO apt-get update -qq && $SUDO apt-get install -y -qq tmux; fi
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
nvidia-smi

if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
source .venv/bin/activate
uv pip install torch==2.15.0 transformers==5.24.1 huggingface_hub==2.4.0 tokenizers==0.23.3 \
  safetensors==0.8.0 numpy==2.5.5 pandas==3.1.1 pyarrow==26.0.1 scipy==1.18.1 matplotlib==3.11.2

cuda_ok() { python -c "import torch; assert torch.cuda.is_available(); torch.ones(1).cuda()" 2>/dev/null; }
if ! cuda_ok; then
  # The default wheel's CUDA build needs a newer driver than this box has: try older builds.
  for cu in cu128 cu126; do
    echo "CUDA not usable with the default torch wheel; trying $cu"
    uv pip install --reinstall torch==2.15.0 --index-url "https://download.pytorch.org/whl/$cu" && cuda_ok && break
  done
fi
cuda_ok || { echo "STOPPED: torch cannot use the GPU (driver too old for torch 2.15.0?)."; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"

python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt']))"
df -h "$REPO" | tail -1

tmux kill-session -t sg2 2>/dev/null || true
tmux new-session -d -s sg2 "SG2_PIPELINE=1 bash '$REPO/sg2/setup_pod.sh' '$PROMPTS' '$RUN' 2>&1 | tee -a '$LOG'; echo '[pipeline exited; press Enter]'; read"
echo "Started in tmux session 'sg2'. Watch: tmux attach -t sg2   or   tail -f $LOG"
