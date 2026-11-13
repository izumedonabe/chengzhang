"""Drive the Python (MLX) engine like `chengzhang --simulate` does and print one JSON line
per file. (The CLI itself currently fails at startup; this calls chengzhang.engine directly
with the same feeding logic, so the comparison uses the Python app's real pipeline.)

    .venv/bin/python native/tools/py_sim.py a.wav b.wav
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chengzhang.engine import SR, Config, Engine  # noqa: E402

ready, done, capture_done = threading.Event(), threading.Event(), threading.Event()
result = {}


def on_event(kind, payload):
    if kind == "ready":
        print(f"loaded in {payload['load_s']:.2f}s", file=sys.stderr, flush=True)
        ready.set()
    elif kind == "capture_done":
        capture_done.set()
    elif kind == "done":
        payload.pop("audio", None)
        result.update(payload)
        done.set()
    elif kind == "error":
        print("error:", payload, file=sys.stderr, flush=True)
        ready.set()
        done.set()


eng = Engine(Config(dictionary=[]), on_event)
eng.start()
ready.wait()
for path in sys.argv[1:]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    done.clear()
    capture_done.clear()
    result.clear()
    eng.send("start")
    stream = np.concatenate([wav, np.zeros(SR, np.float32)])
    t0, released = time.perf_counter(), False
    for i in range(0, len(stream), 512):
        if not released and i >= len(wav):
            eng.send("end", time.perf_counter())
            released = True
        if released and capture_done.is_set():
            break
        eng.send("audio", stream[i : i + 512])
        due = t0 + (i + 512) / SR
        time.sleep(max(0.0, due - time.perf_counter()))
    eng.send("flush")
    done.wait()
    print(json.dumps({"file": Path(path).stem, **{k: result.get(k) for k in (
        "text", "raw", "release_to_raw_s", "release_to_first_char_s", "release_to_done_s")}},
        ensure_ascii=False), flush=True)
eng.send("quit")
