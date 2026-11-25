"""Self-running v2 job for a Colab GPU runtime, executed as one kernel cell:

    colab upload -s S sg2_job.tgz /content/sg2_job.tgz
    colab drivemount -s S
    colab exec -s S --timeout 0 -f sg2/colab_job.py

It first checks that Google Drive is mounted and writable (and stops loudly if not), then
unpacks the code and runs sg2/setup_pod.sh under supervision (installs pinned packages,
downloads the model, starts the chain in tmux), then watches the chain: every SYNC_MIN minutes
it copies the whole run folder (results only; no model weights or packages) to a Drive folder
that is new for every launch, SG2_v2_results/<run>_<timestamp>. The chain's real exit code is
read from runs/<run>.exit, which setup_pod.sh writes when the chain ends. When the chain ends
the job copies once more, verifies the copy (every file present on Drive with the same size),
and releases the GPU runtime itself, so nothing depends on the laptop.

Guards, each of which stops the chain, makes a verified final copy and then releases the GPU:
- setup (or a bad tarball) failed, or setup ran over SETUP_LIMIT_MIN;
- chain failed: waits FAIL_GRACE_MIN so someone can take over (touch /content/KEEP to keep
  the runtime), then releases;
- no new log output for STALL_MIN minutes;
- total time over LIMIT_H hours (setup, run, copies and the grace period all count).
The runtime is never released before the final copy is verified, except as the very last step
once the total time limit has passed and the copy still cannot be verified.
"""
import os, shutil, signal, subprocess, time, traceback
from pathlib import Path

RUN_NAME = os.environ.get("SG2_RUN", "v2")
WORK = Path("/content/sg2v2")
RUN = WORK / "runs" / RUN_NAME
LOG = WORK / "runs" / f"{RUN_NAME}.log"
EXIT = WORK / "runs" / f"{RUN_NAME}.exit"
SETUP_LOG = WORK / "setup.log"
DRIVE = Path("/content/drive/MyDrive/SG2_v2_results") / f"{RUN_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
SYNC_MIN, STALL_MIN, LIMIT_H, FAIL_GRACE_MIN, SETUP_LIMIT_MIN = 5, 40, 6, 20, 30
T0 = time.time()
DEADLINE = T0 + LIMIT_H * 3600


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:.0f}m] {msg}", flush=True)


def copy_to_drive():
    """Copy the run folder and the logs to Drive. Never raises; returns True if every copy
    step reported success."""
    ok = True
    try:
        DRIVE.mkdir(parents=True, exist_ok=True)
        if RUN.exists():
            done = False
            try:
                r = subprocess.run(["rsync", "-a", f"{RUN}/", f"{DRIVE}/run/"],
                                   capture_output=True, text=True, timeout=900)
                done = r.returncode == 0
                if not done:
                    say(f"rsync exit {r.returncode}: {r.stderr.strip()[-300:]}; using shutil")
            except Exception as e:  # noqa: BLE001  (rsync missing, or timed out)
                say(f"rsync unavailable ({e!r}); using shutil")
            if not done:
                shutil.copytree(RUN, DRIVE / "run", dirs_exist_ok=True)
        for f in (LOG, SETUP_LOG, EXIT):
            if f.exists():
                shutil.copy2(f, DRIVE / f.name)
    except Exception as e:  # noqa: BLE001
        say(f"WARNING: copy to Drive failed: {e!r}")
        ok = False
    return ok


def sizes(root):
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = Path(d) / f
            try:
                out[str(p.relative_to(root))] = p.stat().st_size
            except OSError:
                pass
    return out


def verify():
    """Every local result file and log is on Drive with the same size. Returns (ok, text)."""
    try:
        local = sizes(RUN) if RUN.exists() else {}
        remote = sizes(DRIVE / "run") if (DRIVE / "run").exists() else {}
        bad = [k for k, v in local.items() if remote.get(k) != v]
        for f in (LOG, SETUP_LOG, EXIT):
            if f.exists() and not ((DRIVE / f.name).exists()
                                   and (DRIVE / f.name).stat().st_size == f.stat().st_size):
                bad.append(f.name)
        return (not bad, f"{len(local)} files, {sum(local.values()) / 1e6:.1f} MB, "
                         f"{len(bad)} missing or different on Drive {bad[:5]}")
    except Exception as e:  # noqa: BLE001
        return False, f"verification error {e!r}"


def sync(note=""):
    """Periodic copy while the chain is running (files may change underneath; not verified)."""
    ok = copy_to_drive()
    say(f"synced to Drive {DRIVE.name}" if ok else "WARNING: this sync did not complete")
    return ok


def stop_pipeline():
    subprocess.run(["tmux", "kill-session", "-t", "sg2"], capture_output=True)
    time.sleep(5)  # let the chain's processes exit so no file changes during the final copy


def release(note):
    """Stop the chain, make a verified final copy, then release the GPU runtime. If the copy
    cannot be verified, keep retrying until the total time limit (at least 15 minutes), and
    release only then."""
    try:
        stop_pipeline()
    except Exception as e:  # noqa: BLE001
        say(f"could not stop tmux session: {e!r}")
    give_up = max(DEADLINE, time.time() + 15 * 60)
    while True:
        copy_to_drive()
        ok, text = verify()
        if ok:
            try:
                (DRIVE / "STATUS.txt").write_text(
                    f"{note} at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\nverified: {text}\n")
            except Exception as e:  # noqa: BLE001
                say(f"could not write STATUS.txt: {e!r}")
            say(f"final copy verified on Drive {DRIVE}: {text}")
            break
        say(f"final copy NOT verified ({text}); retrying in 60 s")
        if time.time() > give_up:
            say("TIME LIMIT passed and the Drive copy is still unverified: releasing as last resort")
            break
        time.sleep(60)
    try:  # push everything from the local Drive cache to Google before the VM disappears
        from google.colab import drive
        drive.flush_and_unmount(timeout_ms=15 * 60 * 1000)
        say("Drive flushed")
    except Exception as e:  # noqa: BLE001
        say(f"Drive flush failed ({e!r}); waiting 3 min for background upload")
        time.sleep(180)
    say(f"releasing GPU runtime: {note}")
    from google.colab import runtime
    runtime.unassign()


def tail(n=3):
    try:
        return "\n    ".join(LOG.read_text(errors="replace").splitlines()[-n:])
    except Exception:  # noqa: BLE001
        return "(no log yet)"


def exit_code():
    """The chain's exit code from the status file setup_pod.sh writes, or None if still running."""
    try:
        return int(EXIT.read_text().strip())
    except Exception:  # noqa: BLE001
        return None


def tmux_alive():
    try:
        return subprocess.run(["tmux", "has-session", "-t", "sg2"], capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def check_drive():
    """Stop before any work if results could not be saved."""
    if not os.path.ismount("/content/drive"):
        raise SystemExit("STOPPED before starting: Google Drive is not mounted at /content/drive. "
                         "Mount it (colab drivemount) and launch again. Nothing was run.")
    try:
        DRIVE.mkdir(parents=True, exist_ok=False)
        probe = DRIVE / "STATUS.txt"
        probe.write_text(f"STARTED at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        assert probe.read_text().startswith("STARTED")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"STOPPED before starting: cannot write to Drive folder {DRIVE}: {e!r}. "
                         "Nothing was run.")
    say(f"Drive ok; results will be copied to {DRIVE}")


def setup():
    """Unpack and run setup_pod.sh with a time limit. Returns None on success, else the reason."""
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "runs").mkdir(exist_ok=True)
    EXIT.unlink(missing_ok=True)
    try:
        r = subprocess.run(["tar", "xzf", "/content/sg2_job.tgz", "-C", str(WORK)],
                           capture_output=True, text=True, timeout=600)
    except Exception as e:  # noqa: BLE001
        return f"could not unpack /content/sg2_job.tgz: {e!r}"
    if r.returncode != 0:
        return f"could not unpack /content/sg2_job.tgz: {r.stderr.strip()[-300:]}"
    say("setup starting (packages, model download)")
    limit = min(T0 + SETUP_LIMIT_MIN * 60, DEADLINE)
    with open(SETUP_LOG, "w") as fh:
        p = subprocess.Popen(["bash", "sg2/setup_pod.sh", "data/accepted_candidates.jsonl",
                              f"runs/{RUN_NAME}"], cwd=WORK, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        while p.poll() is None:
            if time.time() > limit:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    p.kill()
                p.wait()
                return f"setup did not finish within {SETUP_LIMIT_MIN} min"
            time.sleep(5)
    try:
        say("setup log tail:\n    " + "\n    ".join(SETUP_LOG.read_text(errors="replace").splitlines()[-6:]))
    except Exception:  # noqa: BLE001
        pass
    return None if p.returncode == 0 else f"setup failed (exit {p.returncode})"


def watch():
    last_size, last_change, last_sync = -1, time.time(), time.time()
    while True:
        time.sleep(30)
        try:
            size = LOG.stat().st_size
        except OSError:
            size = 0
        if size != last_size:
            last_size, last_change = size, time.time()
        alive = tmux_alive()
        code = exit_code()
        if code is not None:
            ok = code == 0 and (RUN / ".done_checks_in_run").exists()
            return release("FINISHED OK") if ok else fail(f"chain stopped with an error (exit {code})")
        if not alive:
            return fail("the tmux session ended without writing an exit status")
        if time.time() - last_sync > SYNC_MIN * 60:
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader"], capture_output=True, text=True,
                                     timeout=60).stdout.strip()
            except Exception as e:  # noqa: BLE001
                gpu = f"n/a ({e!r})"
            say(f"GPU {gpu}; log:\n    {tail()}")
            sync()
            last_sync = time.time()
        if time.time() - last_change > STALL_MIN * 60:
            return release(f"STALLED: no log output for {STALL_MIN} min")
        if time.time() > DEADLINE:
            return release(f"TIME LIMIT {LIMIT_H} h reached")


def fail(why):
    say(f"FAILED: {why}. Log:\n    {tail(15)}")
    sync()
    try:
        (DRIVE / "STATUS.txt").write_text(f"FAILED: {why} at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    except Exception:  # noqa: BLE001
        pass
    grace_end = min(time.time() + FAIL_GRACE_MIN * 60, max(DEADLINE, time.time()))
    say(f"waiting up to {FAIL_GRACE_MIN} min; touch /content/KEEP to keep the runtime")
    while time.time() < grace_end:
        time.sleep(30)
        if Path("/content/KEEP").exists():
            say("KEEP found: runtime left running for manual takeover")
            return
    release(f"FAILED: {why}")


def main():
    check_drive()  # raises before anything runs if results could not be saved
    try:
        why = setup()
        if why:
            return fail(why)
        watch()
    except Exception:  # noqa: BLE001  (a watcher bug must not leave the GPU unwatched)
        fail("watcher error: " + traceback.format_exc()[-800:])


main()
