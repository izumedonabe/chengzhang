"""Headless test of idle model unloading: unload → dictate (auto-wake) → unload all → wake.
Wav file instead of mic, stubbed paste, temp data dir.

usage: python bench/unload_test.py data/dictation/d03.wav
"""
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import app_smoke as S  # reuses its stubs (paste, callAfter) and FileMic
from chengzhang.app import controller as C
from chengzhang.app.store import Store


def wait(ctl, states, timeout=60):
    t = time.time()
    while ctl.status["state"] not in states and time.time() - t < timeout:
        time.sleep(0.05)
    return time.time() - t


def dictate(ctl, store, wav, sr):
    n = len(store.history(100))
    t0 = time.time()
    ctl._hotkey_down(False)
    started = ctl.recording
    time.sleep(len(wav) / sr - 0.3)
    ctl._hotkey_up()
    while len(store.history(100)) == n and time.time() - t0 < 30:
        time.sleep(0.05)
    e = store.history(1)[0]
    return started, e


def main():
    wav, sr = sf.read(sys.argv[1], dtype="float32")
    store = Store(Path(tempfile.mkdtemp()))
    ctl = C.Controller(store)
    ctl.mic = S.FileMic(ctl._on_audio)
    ctl.mic.wav = wav
    hud = []
    def fake_hud(state, payload=None):  # record, but keep the controller's bookkeeping
        hud.append(state)
        if state != "hidden":
            ctl._hud_gen += 1
    ctl.hud = fake_hud
    ctl.load_engine()
    wait(ctl, ("ready", "error"))
    print(f"loaded            mem {ctl.status.get('memory_gb')} GB")

    started, e = dictate(ctl, store, wav, sr)
    print(f"1 normal          polished={e['polished']} done {e['done_s']:.2f}s  {e['text']}")

    ctl.unload("llm")
    wait(ctl, ("asleep",))
    print(f"unload llm        state={ctl.status['state']} mem {ctl.status.get('memory_gb')} GB")

    hud.clear()
    started, e = dictate(ctl, store, wav, sr)
    print(f"2 after unload    listened immediately={started} polished={e['polished']} "
          f"done {e['done_s']:.2f}s  hud: {' → '.join(dict.fromkeys(hud))}")
    print(f"                  {e['text']}")
    wait(ctl, ("ready",))
    print(f"                  state={ctl.status['state']} mem {ctl.status.get('memory_gb')} GB")

    ctl.unload("all")
    wait(ctl, ("asleep",))
    print(f"unload all        state={ctl.status['state']} mem {ctl.status.get('memory_gb')} GB")
    hud.clear()
    ctl._hotkey_down(False)
    print(f"3 press while all unloaded: listening={ctl.recording} hud={hud}")
    dt = wait(ctl, ("ready",))
    print(f"                  woke in {dt:.1f}s, mem {ctl.status.get('memory_gb')} GB")
    started, e = dictate(ctl, store, wav, sr)
    print(f"4 after wake      polished={e['polished']} done {e['done_s']:.2f}s  {e['text']}")


if __name__ == "__main__":
    main()
