"""Headless integration test of the app controller: wav file instead of mic,
direct calls instead of the hotkey, stubbed paste. Uses a temp data dir.

usage: python bench/app_smoke.py data/dictation/d03.wav [--translate]
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from chengzhang.app import controller as C
from chengzhang.app import macos
from chengzhang.app.store import Store

pasted = []
C.AppHelper.callAfter = lambda f, *a: f(*a)  # no AppKit run loop here
macos.paste_text = lambda t: pasted.append(t)
macos.frontmost_app = lambda: "Notes"
C.macos = macos


class FileMic:
    """Stands in for the microphone: plays a wav in real time, then silence."""

    def __init__(self, on_audio):
        self.on_audio, self.level, self.stream = on_audio, 0.0, None
        self.capturing, self.name = False, "file"

    def configure(self, device, keep_open):
        pass

    def start(self):
        self.stream = True
        self.capturing = True
        wav = self.wav

        def run():
            t0 = time.perf_counter()
            for i in range(0, len(wav) + 16000, 512):
                if self.stream is None:
                    return
                self.on_audio(wav[i : i + 512] if i < len(wav) else np.zeros(512, np.float32))
                time.sleep(max(0, t0 + (i + 512) / 16000 - time.perf_counter()))

        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.stream = None
        self.capturing = False


def main():
    path = sys.argv[1]
    translate = "--translate" in sys.argv
    wav, sr = sf.read(path, dtype="float32")
    store = Store(Path(tempfile.mkdtemp()))
    store.update_settings(save_audio=True)
    if "--llm" in sys.argv:
        store.update_settings(llm_model=sys.argv[sys.argv.index("--llm") + 1])
    ctl = C.Controller(store)
    ctl.mic = FileMic(ctl._on_audio)
    ctl.mic.wav = wav
    hud = []
    ctl.hud = lambda s, p=None: hud.append(s)
    ctl.load_engine()
    while ctl.status["state"] not in ("ready", "error"):
        time.sleep(0.2)
    assert ctl.status["state"] == "ready", ctl.status

    if "--toggle" in sys.argv:
        store.update_settings(hotkey_mode="toggle")
        ctl._hotkey_down(translate)          # first press: start
        ctl._hotkey_up()                     # releasing does nothing in toggle mode
        assert ctl.recording, "toggle: release must not stop"
        time.sleep(len(wav) / sr - 0.3)
        ctl._hotkey_down(False)              # second press: stop
    else:
        ctl._hotkey_down(translate)
        time.sleep(len(wav) / sr - 0.3)  # let go 300 ms before the speech ends
        ctl._hotkey_up()
    t = time.time()
    while not store.history(1) and time.time() - t < 10:
        time.sleep(0.05)
    e = store.history(1)[0]
    print("pasted :", pasted)
    print("history:", {k: e[k] for k in ("text", "raw", "app", "mode", "audio_s", "done_s")})
    print("hud    :", " → ".join(dict.fromkeys(hud)))
    print("stats  :", store.stats()["today"])
    assert pasted and pasted[0] == e["text"]
    wav_path = store.root / "recordings" / f"{e['id']}.wav"
    print("audio  :", wav_path.name, f"{sf.info(wav_path).duration:.1f}s" if wav_path.exists() else "MISSING")


if __name__ == "__main__":
    main()
