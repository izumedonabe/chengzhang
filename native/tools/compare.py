"""Compare the Python (MLX) and native (llama.cpp) engines on the same WAV files.

Runs both simulators alternately (the machine's state drifts: memory pressure,
GPU clocks), `rounds` times each, and reports per-file medians.

    .venv/bin/python -I native/tools/compare.py data/dictation/*.wav --rounds 2
"""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT.parent.parent.parent / ".venv/bin/python"  # worktree -> main checkout's venv
if not PY.exists():
    PY = Path(sys.executable)
SIM = ROOT / "native/target/release/cz-sim"


def run_python(wavs):
    out = subprocess.run([str(PY), str(ROOT / "native/tools/py_sim.py"), *wavs],
                         cwd=ROOT, capture_output=True, text=True)
    res = {}
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            res[r["file"]] = {"first": r["release_to_first_char_s"], "done": r["release_to_done_s"],
                              "raw": r["release_to_raw_s"], "text": r["text"]}
    return res


def run_native(wavs):
    out = subprocess.run([str(SIM), *wavs], capture_output=True, text=True)
    res, name = {}, None
    for line in out.stdout.splitlines():
        if line.startswith("== "):
            name = line.split()[1]
            res[name] = {}
        elif "text:" in line:
            res[name]["text"] = line.split("text:", 1)[1].strip().replace("⏎", "\n")
        elif "release→raw" in line:
            m = re.search(r"release→raw (\d+)ms\s+→first char (\S+)\s+→done (\d+)ms", line)
            res[name]["raw"] = int(m[1]) / 1e3
            res[name]["first"] = None if m[2] == "-" else int(m[2].rstrip("ms")) / 1e3
            res[name]["done"] = int(m[3]) / 1e3
    return res


def main():
    args = sys.argv[1:]
    rounds = 2
    if "--rounds" in args:
        i = args.index("--rounds")
        rounds = int(args[i + 1])
        del args[i:i + 2]
    wavs = [str(Path(a).resolve()) for a in args]
    runs = {"python": [], "native": []}
    for r in range(rounds):
        for name, fn in (("python", run_python), ("native", run_native)):
            print(f"round {r + 1}: {name}…", file=sys.stderr, flush=True)
            runs[name].append(fn(wavs))
    med = lambda xs: statistics.median(x for x in xs if x is not None) if any(x is not None for x in xs) else None
    fmt = lambda x: "-" if x is None else f"{x * 1e3:.0f}"
    print(f"{'file':8} | {'松开→首字 py / native (ms)':>26} | {'松开→完成 py / native (ms)':>26} | same text")
    for w in wavs:
        n = Path(w).stem
        p = [r.get(n, {}) for r in runs["python"]]
        q = [r.get(n, {}) for r in runs["native"]]
        same = p[-1].get("text") == q[-1].get("text")
        print(f"{n:8} | {fmt(med([x.get('first') for x in p])):>12} / {fmt(med([x.get('first') for x in q])):<11} | "
              f"{fmt(med([x.get('done') for x in p])):>12} / {fmt(med([x.get('done') for x in q])):<11} | {'yes' if same else 'no'}")
    print("\nlast-round texts that differ:")
    for w in wavs:
        n = Path(w).stem
        a, b = runs["python"][-1].get(n, {}).get("text"), runs["native"][-1].get(n, {}).get("text")
        if a != b:
            print(f"  {n}\n    py    : {a!r}\n    native: {b!r}")


if __name__ == "__main__":
    main()
