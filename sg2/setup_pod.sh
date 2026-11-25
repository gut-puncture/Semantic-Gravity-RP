#!/usr/bin/env bash
# Set up a plain Linux GPU box (e.g. Prime Intellect pod, Ubuntu + CUDA image) and run the
# whole v2 job inside tmux (chain in sg2/pipeline.py): probe_p0 -> select -> checks ->
# 64-prompt smoke test -> full run with patching -> analyze.
#
#   bash sg2/setup_pod.sh data/accepted_candidates.jsonl runs/v2
#   SG2_PROMPTS=data/prompts_v2.jsonl bash sg2/setup_pod.sh - runs/v2    # skip probe+select
#   SG2_PER_CAT=2 ...                                                   # tiny selection (tests)
#
# Run from the repo root. Everything is logged to <run-dir>.log. The job runs in tmux
# session "sg2"; detach with Ctrl-b d, reattach with `tmux attach -t sg2`. Rerunning the
# script resumes: finished steps are skipped, and the probe and the experiment skip finished
# work. It stops with a clear message at the first failed step.
set -euo pipefail

CANDIDATES="${1:?usage: setup_pod.sh <accepted_candidates.jsonl> <run-dir>}"
RUN="${2:?usage: setup_pod.sh <accepted_candidates.jsonl> <run-dir>}"
MODEL="${SG2_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO" HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
LOG="$RUN.log"
INPUT=(--candidates "$CANDIDATES")
[[ -n "${SG2_PROMPTS:-}" ]] && INPUT=(--prompts "$SG2_PROMPTS")
EXTRA=()
[[ -n "${SG2_PER_CAT:-}" ]] && EXTRA+=(--per-cat "$SG2_PER_CAT")

# ---------------------------------------------------------------- pipeline (runs in tmux)
if [[ "${SG2_PIPELINE:-}" == "1" ]]; then
  # Whatever ends this branch, record the real exit code in <run-dir>.exit (written atomically);
  # sg2/colab_job.py reads it. The tmux pane's own text never reaches the log.
  trap 'rc=$?; echo "$rc" > "$RUN.exit.tmp" && mv "$RUN.exit.tmp" "$RUN.exit"' EXIT
  source .venv/bin/activate
  nvidia-smi || true
  python -u -m sg2.pipeline --run-dir "$RUN" --model "$MODEL" "${INPUT[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
  exit $?
fi

# ---------------------------------------------------------------- setup (runs once)
f="${SG2_PROMPTS:-$CANDIDATES}"; [[ -f "$f" ]] || { echo "input file not found: $f"; exit 1; }
SUDO=""; [[ $(id -u) -ne 0 ]] && SUDO="sudo"
if ! command -v tmux >/dev/null; then $SUDO apt-get update -qq && $SUDO apt-get install -y -qq tmux; fi
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[[ -n "${SG2_CPU_TEST:-}" ]] || nvidia-smi   # SG2_CPU_TEST=1: rehearse the job on a CPU machine

if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
source .venv/bin/activate
uv pip install torch==2.15.0 transformers==5.24.1 huggingface_hub==2.4.0 tokenizers==0.23.3 \
  safetensors==0.8.0 numpy==2.5.5 pandas==3.1.1 pyarrow==26.0.1 scipy==1.18.1 matplotlib==3.11.2

cuda_ok() { [[ -n "${SG2_CPU_TEST:-}" ]] && return 0; python -c "import torch; assert torch.cuda.is_available(); torch.ones(1).cuda()" 2>/dev/null; }
if ! cuda_ok; then
  # The default wheel's CUDA build needs a newer driver than this box has: try older builds.
  for cu in cu128 cu126; do
    echo "CUDA not usable with the default torch wheel; trying $cu"
    uv pip install --reinstall torch==2.15.0 --index-url "https://download.pytorch.org/whl/$cu" && cuda_ok && break
  done
fi
cuda_ok || { echo "STOPPED: torch cannot use the GPU (driver too old for torch 2.15.0?)."; exit 1; }
[[ -n "${SG2_CPU_TEST:-}" ]] || python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"

python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt']))"
df -h "$REPO" | tail -1

[[ -n "${SG2_SETUP_ONLY:-}" ]] && { echo "setup only: done"; exit 0; }
tmux kill-session -t sg2 2>/dev/null || true
mkdir -p "$(dirname "$LOG")"; rm -f "$RUN.exit" "$RUN.exit.tmp"
tmux new-session -d -s sg2 "SG2_PIPELINE=1 SG2_MODEL='$MODEL' SG2_PROMPTS='${SG2_PROMPTS:-}' SG2_PER_CAT='${SG2_PER_CAT:-}' bash '$REPO/sg2/setup_pod.sh' '$CANDIDATES' '$RUN' 2>&1 | tee -a '$LOG'; echo '[pipeline exited; press Enter]'; read"
echo "Started in tmux session 'sg2'. Watch: tmux attach -t sg2   or   tail -f $LOG"
