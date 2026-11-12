"""Transcribe the TTS dictation clips -> bench/asr_transcripts.json."""
import json
import sys
import time

from mlx_audio.stt import load

model = load(sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-ASR-0.6B-8bit")
ids = [l.split("\t")[0] for l in open("data/dictation/scripts.tsv", encoding="utf-8")]
model.generate(f"data/dictation/{ids[0]}.wav", language="Chinese")  # warmup
out = {}
for i in ids:
    t = time.perf_counter()
    text = model.generate(f"data/dictation/{i}.wav", language="Chinese").text
    print(f"{i} {time.perf_counter() - t:.3f}s {text}")
    out[i] = text
json.dump(out, open("bench/asr_transcripts.json", "w"), ensure_ascii=False, indent=1)
