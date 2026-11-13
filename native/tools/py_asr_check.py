"""Reference for crates/cz-mlx/examples/asr_check: mlx-audio Qwen3-ASR, language=Chinese."""
import sys, time
import soundfile as sf
from mlx_audio.stt import load
m = load(sys.argv[1])
for f in sys.argv[2:]:
    wav, _ = sf.read(f, dtype="float32")
    for r in range(2):
        time.sleep(float(__import__("os").environ.get("IDLE_MS", "0")) / 1000)
        t = time.perf_counter()
        s = m.generate(wav, language="Chinese").text.strip()
        print(f"{f.rsplit('/',1)[-1]} round {r}: {(time.perf_counter()-t)*1e3:.0f} ms  {s}")
