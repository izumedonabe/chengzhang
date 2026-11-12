"""Side-by-side on our own inputs: the 8 TTS dictation transcripts (real ASR
errors) + the user's real dictations from the app history.

usage: python bench/compare_polishers.py refiner qwen35-4b
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import aasr_polish_eval as ev  # noqa: E402

inputs = list(json.load(open(ROOT / "bench/asr_transcripts.json", encoding="utf-8")).values())
hist = Path(os.path.expanduser("~/Library/Application Support/OpenTypeLess/history.jsonl"))
if hist.exists():
    inputs += [json.loads(l)["raw"] for l in hist.read_text(encoding="utf-8").splitlines() if l.strip()]

results = {}
for name in sys.argv[1:]:
    pol = ev.make_polisher(name)
    pol("嗯那个我们明天开会吧")
    results[name] = []
    for x in inputs:
        r = pol(x)
        results[name].append((ev.clean_output(r["text"]), r["total_s"]))
    del pol
    import importlib

    from chengzhang import prompts
    importlib.reload(prompts)  # undo the refiner's prompt patch for the next model

for i, x in enumerate(inputs):
    print(f"\n[{i + 1}] 原始：{x}")
    for name in results:
        text, s = results[name][i]
        print(f"    {name:10s} {s:4.2f}s  {text}")
