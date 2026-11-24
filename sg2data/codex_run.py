"""Run LLM work units through Codex (ChatGPT plan), one `codex exec` call per unit.

Each call is read-only (the model cannot run code or write files, so it must answer each
item itself) and constrained by the unit's JSON Schema via --output-schema. Output goes to
data/llm_outputs/<kind>/<unit_id>.json, which sg2data.llm_collect reads. Finished units are
skipped, so rerunning the same command resumes.

We first tried one gpt-6.2-sol orchestrator spawning worker subagents. In the pilot the
judge workers produced fabricated labels (the question's subject word as the answer,
duplicated top-3 lists), consistent with writing a script instead of judging. Direct
read-only calls removed this.

Usage: python3 -m sg2data.codex_run --kind gen|judge [--glob unit_*.json] [--parallel 12]
"""
import argparse, json, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNITS = ROOT / "data" / "llm_units"
OUT = ROOT / "data" / "llm_outputs"
MODEL = {"gen": "gpt-6.1-terra", "judge": "gpt-6-luna"}
EFFORT = {"gen": "medium", "judge": "medium"}
PREFIX = ("Work only by reading and thinking. Do not run any commands, write code or create "
          "files; produce every answer yourself.\n\n")


def valid(p):
    try:
        json.loads(p.read_text())
        return True
    except Exception:
        return False


def run_unit(kind, unit_file, tmp):
    unit = json.loads(unit_file.read_text())
    out = OUT / kind / f"{unit_file.stem}.json"
    schema = Path(tmp) / f"{unit_file.stem}.schema.json"
    schema.write_text(json.dumps(unit["output_schema"]))
    cmd = ["codex", "exec", "-m", MODEL[kind], "-c", f'model_reasoning_effort="{EFFORT[kind]}"',
           "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", str(tmp),
           "--output-schema", str(schema), "-o", str(out), PREFIX + unit["prompt"]]
    t0 = time.time()
    for attempt in range(2):
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
        if out.exists() and valid(out):
            return unit_file.stem, True, time.time() - t0, ""
    return unit_file.stem, False, time.time() - t0, (r.stdout + r.stderr)[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["gen", "judge"], required=True)
    ap.add_argument("--glob", default="unit_*.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=12)
    a = ap.parse_args()
    (OUT / a.kind).mkdir(parents=True, exist_ok=True)
    todo = [f for f in sorted((UNITS / a.kind).glob(a.glob))
            if not ((OUT / a.kind / f"{f.stem}.json").exists() and valid(OUT / a.kind / f"{f.stem}.json"))]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} {a.kind} units to run with {MODEL[a.kind]}", flush=True)
    t0, n_ok, n_bad = time.time(), 0, 0
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(a.parallel) as ex:
        futs = [ex.submit(run_unit, a.kind, f, tmp) for f in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            name, ok, dt, err = fu.result()
            n_ok += ok; n_bad += (not ok)
            if not ok:
                print(f"FAILED {name}: {err}", flush=True)
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)} done, {n_ok} ok, {n_bad} failed, {el/60:.1f} min, "
                      f"eta {(len(todo)-i)*el/i/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
