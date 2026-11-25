"""Self-running v2 job for a Colab GPU runtime, executed as one kernel cell:

    colab upload -s S sg2_job.tgz /content/sg2_job.tgz
    colab drivemount -s S
    colab exec -s S --timeout 0 -f sg2/colab_job.py

It unpacks the code, runs sg2/setup_pod.sh (installs pinned packages, downloads the model,
starts the chain in tmux), then watches the chain: every SYNC_MIN minutes it copies the whole
run folder (results only; no model weights or packages) to Google Drive. When the chain ends
it copies once more and releases the GPU runtime itself, so nothing depends on the laptop.

Guards, each of which also syncs and then releases the GPU:
- chain failed: waits FAIL_GRACE_MIN so someone can take over (touch /content/KEEP to keep
  the runtime), then releases;
- no new log output for STALL_MIN minutes;
- total time over LIMIT_H hours.
"""
import os, shutil, subprocess, time
from pathlib import Path

RUN_NAME = os.environ.get("SG2_RUN", "v2")
WORK = Path("/content/sg2v2")
RUN = WORK / "runs" / RUN_NAME
LOG = WORK / "runs" / f"{RUN_NAME}.log"
DRIVE = Path("/content/drive/MyDrive/SG2_v2_results") / RUN_NAME
SYNC_MIN, STALL_MIN, LIMIT_H, FAIL_GRACE_MIN = 5, 40, 5, 20
T0 = time.time()


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.0f}m] {msg}", flush=True)


def sync(note=""):
    if not os.path.ismount("/content/drive"):
        say("WARNING: Drive not mounted; results stay on the VM only")
        return
    DRIVE.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["rsync", "-a", f"{RUN}/", f"{DRIVE}/run/"], capture_output=True, text=True)
    if r.returncode != 0:
        shutil.copytree(RUN, DRIVE / "run", dirs_exist_ok=True)
    for f in (LOG, WORK / "setup.log"):
        if f.exists():
            shutil.copy2(f, DRIVE / f.name)
    if note:
        (DRIVE / "STATUS.txt").write_text(f"{note} at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    say(f"synced to Drive {note}")


def release(note):
    sync(note)
    say(f"releasing GPU runtime: {note}")
    from google.colab import runtime
    runtime.unassign()


def tail(n=3):
    try:
        return "\n    ".join(LOG.read_text().splitlines()[-n:])
    except FileNotFoundError:
        return "(no log yet)"


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "xzf", "/content/sg2_job.tgz", "-C", str(WORK)], check=True)
    (WORK / "runs").mkdir(exist_ok=True)
    say("setup starting (packages, model download)")
    with open(WORK / "setup.log", "w") as fh:
        r = subprocess.run(["bash", "sg2/setup_pod.sh", "data/accepted_candidates.jsonl", f"runs/{RUN_NAME}"],
                           cwd=WORK, stdout=fh, stderr=subprocess.STDOUT)
    say("setup log tail:\n    " + "\n    ".join((WORK / "setup.log").read_text().splitlines()[-6:]))
    if r.returncode != 0:
        RUN.mkdir(parents=True, exist_ok=True)
        return fail("setup failed")
    last_size, last_change, last_sync = -1, time.time(), time.time()
    while True:
        time.sleep(60)
        size = LOG.stat().st_size if LOG.exists() else 0
        if size != last_size:
            last_size, last_change = size, time.time()
        text = LOG.read_text() if LOG.exists() else ""
        if "[pipeline exited" in text:
            ok = (RUN / ".done_checks_in_run").exists()
            return release("FINISHED OK") if ok else fail("chain stopped with an error")
        if time.time() - last_sync > SYNC_MIN * 60:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                  "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
            say(f"GPU {gpu}; log:\n    {tail()}")
            sync()
            last_sync = time.time()
        if time.time() - last_change > STALL_MIN * 60:
            return release(f"STALLED: no log output for {STALL_MIN} min")
        if time.time() - T0 > LIMIT_H * 3600:
            return release(f"TIME LIMIT {LIMIT_H} h reached")


def fail(why):
    say(f"FAILED: {why}. Log:\n    {tail(15)}")
    sync(f"FAILED: {why}")
    say(f"waiting {FAIL_GRACE_MIN} min; touch /content/KEEP to keep the runtime")
    for _ in range(FAIL_GRACE_MIN):
        time.sleep(60)
        if Path("/content/KEEP").exists():
            say("KEEP found: runtime left running for manual takeover")
            return
    release(f"FAILED: {why}")


main()
