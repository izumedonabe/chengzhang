"""Qwen3-ASR benchmark on FLEURS cmn_hans_cn test: CER + latency.

usage: python bench/asr_bench.py MODEL_ID [N] [--auto-lang]
"""
import io
import json
import re
import sys
import time
import unicodedata

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from mlx_audio.stt import load

PARQUET = "data/fleurs/test.parquet"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    # drop punctuation / symbols / whitespace, keep letters + digits + CJK
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LN")


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_fleurs(n: int):
    t = pq.read_table(PARQUET, columns=["id", "audio", "transcription", "raw_transcription"])
    rows = t.slice(0, n).to_pylist()
    out = []
    for r in rows:
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == 16000, sr
        out.append((r["id"], wav, r["raw_transcription"]))
    return out


def main():
    model_id = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
    lang = None if "--auto-lang" in sys.argv else "Chinese"

    t0 = time.perf_counter()
    model = load(model_id)
    load_s = time.perf_counter() - t0
    data = load_fleurs(n)

    # warmup (Metal kernel compile)
    model.generate(data[0][1], language=lang)
    model.generate(data[0][1][: 16000 * 2], language=lang)

    errs = refs = 0
    errs_c = refs_c = 0  # subset without digits / latin letters
    empty = 0
    lat, dur = [], []
    for _id, wav, ref in data:
        t = time.perf_counter()
        out = model.generate(wav, language=lang)
        mx.eval(mx.array(0))
        dt = time.perf_counter() - t
        hyp = out.text
        if not hyp.strip():
            empty += 1
        r, h = norm(ref), norm(hyp)
        errs += edit_distance(r, h)
        refs += len(r)
        if not any(c.isascii() and c.isalnum() for c in r):
            errs_c += edit_distance(r, h)
            refs_c += len(r)
        lat.append(dt)
        dur.append(len(wav) / 16000)

    # short-tail latency: what's left after the user releases the hotkey
    tail = {}
    for sec in (1, 2, 3, 5):
        ts = []
        for _id, wav, _ in data[:20]:
            seg = wav[: 16000 * sec]
            t = time.perf_counter()
            model.generate(seg, language=lang)
            ts.append(time.perf_counter() - t)
        tail[sec] = float(np.median(ts))

    lat, dur = np.array(lat), np.array(dur)
    res = {
        "model": model_id,
        "n": n,
        "language": lang or "auto",
        "load_s": round(load_s, 2),
        "cer_pct": round(100 * errs / refs, 2),
        "cer_no_digits_latin_pct": round(100 * errs_c / refs_c, 2),
        "no_digits_latin_chars": refs_c,
        "empty_outputs": empty,
        "mean_audio_s": round(float(dur.mean()), 2),
        "latency_mean_s": round(float(lat.mean()), 3),
        "latency_p90_s": round(float(np.percentile(lat, 90)), 3),
        "rtf_x_realtime": round(float(dur.sum() / lat.sum()), 1),
        "tail_latency_median_s": {f"{k}s": round(v, 3) for k, v in tail.items()},
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    print(json.dumps(res, ensure_ascii=False))
    with open("bench/results_asr.jsonl", "a") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
