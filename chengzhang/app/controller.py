"""App controller: owns the engine, microphone, hotkey, HUD and storage.
The web UI talks to it through `Api` (pywebview js_api)."""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback

import numpy as np
from PyObjCTools import AppHelper

from ..engine import SR, Config, Engine
from . import macos, models
from .store import Store


BUILTIN_HINTS = ("MacBook", "Built-in", "内建", "内置", "iMac", "Studio Display")
BLUETOOTH_HINTS = ("AirPods", "Beats", "Bluetooth", "蓝牙", "Buds", "WH-", "WF-")


def list_mics() -> list[dict]:
    import sounddevice as sd

    default = sd.query_devices(kind="input")["name"]
    out = []
    for d in sd.query_devices():
        if d["max_input_channels"] > 0:
            n = d["name"]
            out.append({"name": n, "default": n == default,
                        "builtin": any(h in n for h in BUILTIN_HINTS),
                        "bluetooth": any(h in n for h in BLUETOOTH_HINTS)})
    return out


def resolve_mic(pref: str):
    """'auto' = the built-in mic when there is one (AirPods-style headsets need
    ~0.5–1 s to switch into call mode when opened, which eats the first words,
    and their call-mode audio is narrowband). Otherwise a device name."""
    mics = list_mics()
    if pref and pref != "auto":
        for m in mics:
            if m["name"] == pref:
                return m["name"]
    for m in mics:
        if m["builtin"]:
            return m["name"]
    return None  # system default


class Mic:
    """16 kHz mono capture from a chosen device (resampled if needed).
    keep_open: the stream runs while the app runs and the last `preroll_s`
    seconds before the hotkey are kept, so the first syllable is never lost."""

    def __init__(self, on_audio, preroll_s: float = 0.5):
        import collections

        import sounddevice as sd

        self.sd, self.on_audio = sd, on_audio
        self.level = 0.0
        self.stream = None
        self.device = None
        self.keep_open = False
        self.capturing = False
        self.ring = collections.deque(maxlen=max(1, int(preroll_s / 0.032)))
        self.opened_at = None  # when the current stream started delivering

    def configure(self, device, keep_open: bool):
        changed = device != self.device or keep_open != self.keep_open
        self.device, self.keep_open = device, keep_open
        if changed and not self.capturing:
            self._close()
            if keep_open:
                self._open()

    def _open(self):
        sd = self.sd
        dev = self.device
        info = sd.query_devices(dev if dev is not None else sd.default.device[0], "input")
        rate = SR
        try:
            sd.check_input_settings(device=dev, samplerate=SR, channels=1, dtype="float32")
        except Exception:
            rate = int(info["default_samplerate"])
        resample = None
        if rate != SR:
            from ..resample import StreamResampler  # numpy only; keeps scipy out of the app

            resample = StreamResampler(rate, SR)

        def cb(indata, frames, t, status):
            x = indata[:, 0].copy()
            if resample:
                x = resample(x)
            if self.opened_at is None:
                self.opened_at = time.perf_counter()
            rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
            # map roughly -60..-10 dBFS to 0..1, smoothed
            lvl = min(1.0, max(0.0, (20 * np.log10(rms) + 60) / 50))
            self.level = 0.6 * self.level + 0.4 * lvl
            if self.capturing:
                self.on_audio(x)
            elif self.keep_open:
                self.ring.append(x)

        self.opened_at = None
        self.stream = sd.InputStream(device=dev, samplerate=rate, channels=1, dtype="float32",
                                     blocksize=int(rate * 0.032), callback=cb)
        self.stream.start()
        self.name = info["name"]

    def _close(self):
        s, self.stream = self.stream, None
        self.ring.clear()
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    def start(self):
        if self.stream is None:
            self._open()
        else:
            for x in list(self.ring):  # audio from just before the key went down
                self.on_audio(x)
            self.ring.clear()
        self.capturing = True

    def stop(self):
        self.capturing = False
        self.level = 0.0
        if not self.keep_open:
            self._close()


class Controller:
    def __init__(self, store: Store):
        self._rec_evt = threading.Event()
        self.store = store
        self.main_win = None
        self.hud_win = None
        self.engine: Engine | None = None
        self.cfg: Config | None = None
        self.status = {"state": "starting", "stage": None, "progress": 0.0, "error": None}
        self.downloads = models.Downloads(self._on_download_change)
        self.mic = Mic(self._on_audio)
        self.ptt: macos.PushToTalk | None = None
        self.recording = False
        self.session = {}
        self._capture_timer = None
        self._hud_gen = 0
        self.loaded = {"asr": False, "llm": False}  # what's in memory right now
        self.last_used = time.monotonic()
        # pywebview's evaluate_js blocks until the main thread runs the JS, so it
        # must never be called on the main thread (hotkey callbacks run there).
        # All JS pushes go through this thread; hotkey work goes through _work.
        self._js = queue.Queue()
        self._work = queue.Queue()
        threading.Thread(target=self._pump, args=(self._js,), daemon=True, name="otl-js").start()
        threading.Thread(target=self._pump, args=(self._work,), daemon=True, name="otl-work").start()

    @staticmethod
    def _pump(q):
        while True:
            fn, args = q.get()
            try:
                fn(*args)
            except Exception:
                traceback.print_exc()

    # ------------------------------------------------------------------ push to UI

    def _emit(self, win, fn, payload=None):
        if win is not None:
            self._js.put((self._eval, (win, fn, payload)))

    @staticmethod
    def _eval(win, fn, payload):
        win.evaluate_js(f"window.{fn} && window.{fn}({json.dumps(payload, ensure_ascii=False)})")

    def ui(self, kind, payload=None):
        self._emit(self.main_win, "otlEmit", {"kind": kind, "payload": payload})

    def hud(self, state, payload=None):
        if state != "hidden":
            self._hud_gen += 1
        self._emit(self.hud_win, "hudState", {"state": state, "payload": payload})
        if self.hud_win is None or self.hud_win.native is None:
            return
        if state == "hidden":
            AppHelper.callAfter(macos.hide_window, self.hud_win.native)
        else:
            AppHelper.callAfter(macos.show_without_focus, self.hud_win.native)

    def _hide_hud_later(self, delay):
        """Hide the HUD after `delay` s — unless something newer was shown since
        (a stale timer from the last dictation must not hide the next one)."""
        gen = self._hud_gen

        def hide():
            if gen == self._hud_gen:
                self.hud("hidden")
        threading.Timer(delay, hide).start()

    def _set_status(self, **kw):
        self.status.update(kw)
        print(f"[otl] status {self.status['state']} {self.status.get('stage') or ''} "
              f"{self.status.get('progress', 0):.2f} {self.status.get('error') or ''}", flush=True)
        self.ui("status", self.status)

    # ------------------------------------------------------------------ lifecycle

    def boot(self):
        """Runs once the GUI is up (pywebview start callback, background thread)."""
        from . import apply_icon  # AppKit resets the Dock icon when launching finishes: set it again now
        AppHelper.callAfter(apply_icon)
        AppHelper.callAfter(self._install_hotkey)
        AppHelper.callAfter(self._style_main_window)
        self._place_hud()
        threading.Thread(target=self._level_ticker, daemon=True).start()
        threading.Thread(target=self._idle_watch, daemon=True, name="otl-idle").start()
        threading.Thread(target=self._accessibility_watch, daemon=True, name="otl-ax").start()
        self.configure_mic()
        self.load_engine()

    def _style_main_window(self):
        """Modern Mac window: no separate title bar. The painting runs to the top
        edge, traffic lights float over it, and an empty unified toolbar gives the
        taller, inset title area (and native window dragging) of current macOS apps."""
        import AppKit

        w = self.main_win.native
        if w is None:
            return
        w.setTitleVisibility_(AppKit.NSWindowTitleHidden)
        w.setTitlebarAppearsTransparent_(True)
        for b in (AppKit.NSWindowCloseButton, AppKit.NSWindowMiniaturizeButton, AppKit.NSWindowZoomButton):
            btn = w.standardWindowButton_(b)
            if btn is not None:
                btn.setHidden_(False)
        toolbar = AppKit.NSToolbar.alloc().initWithIdentifier_("otl-main")
        toolbar.setShowsBaselineSeparator_(False)
        w.setToolbar_(toolbar)
        w.setToolbarStyle_(AppKit.NSWindowToolbarStyleUnified)
        w.setTitlebarSeparatorStyle_(AppKit.NSTitlebarSeparatorStyleNone)

    def _place_hud(self):
        try:
            x, y, w, h = macos.main_screen_frame()
            hw, hh = 320, 96
            self.hud_win.move(int(x + (w - hw) / 2), int(h - hh - 36))
        except Exception:
            traceback.print_exc()

    def _install_hotkey(self):
        if self.ptt:
            self.ptt.uninstall()
        # the monitor runs on the main thread: only enqueue there, never block
        later = lambda fn: (lambda *a: self._work.put((fn, a)))
        self.ptt = macos.HotkeyTap(self.store.settings()["hotkey"], later(self._hotkey_down),
                                   later(self._hotkey_up), later(self._hotkey_cancel))
        how = self.ptt.install()
        print(f"[otl] hotkey {macos.hotkey_label(self.ptt.spec)} via {how}; "
              f"accessibility={macos.accessibility_trusted(False)}", flush=True)

    def _make_config(self) -> Config:
        s = self.store.settings()
        return Config(
            asr_model=s["asr_model"],
            llm_model=models.resolve_path(s["llm_model"]) or Config.llm_model,
            llm_prompt=models.entry(s["llm_model"]).get("prompt", "default"),
            polish=bool(s["llm_model"]),
            language=None if s["language"] == "auto" else s["language"],
            dictionary=self.store.dictionary(),
            translate_lang=s["translate_lang"] or None,
            fast_path=s["fast_path"],
            post_roll_max_ms=int(s["post_roll_ms"]),
        )

    def load_engine(self):
        s = self.store.settings()
        missing = [m for m in (s["asr_model"], s["llm_model"]) if m and not models.is_downloaded(m)]
        old, self.engine = self.engine, None
        if old:
            old.send("quit")
        if missing:
            self._set_status(state="downloading", stage="download", progress=0.0, error=None)
            for m in missing:
                self.downloads.start(m)
            return
        self.cfg = self._make_config()
        self._set_status(state="loading", stage="vad", progress=0.02, error=None)
        eng = Engine(self.cfg, lambda k, p, e=None: self._on_engine(eng, k, p))
        self.engine = eng
        eng.start()

    def _on_download_change(self, repo):
        self.ui("models", self.catalog())
        if self.status["state"] != "downloading":
            return
        s = self.store.settings()
        needed = [m for m in (s["asr_model"], s["llm_model"]) if m]
        states = [self.downloads.state(m) for m in needed]
        errors = [st["error"] for st in states if st and st["status"] == "error"]
        if errors:
            self._set_status(state="error", error="模型下载失败：" + errors[0])
        elif all(models.is_downloaded(m) for m in needed):
            self.load_engine()

    def download_progress(self) -> float:
        s = self.store.settings()
        needed = [m for m in (s["asr_model"], s["llm_model"]) if m]
        ps = []
        for m in needed:
            st = self.downloads.state(m)
            ps.append(1.0 if models.is_downloaded(m) else (st["progress"] if st else 0.0))
        return sum(ps) / len(ps) if ps else 1.0

    # ------------------------------------------------------------------ engine events

    def _on_engine(self, eng, kind, payload):
        if eng is not self.engine:
            return  # events from an engine we already replaced
        if kind == "loading":
            reload = payload.pop("reload", False)
            if reload:
                self._set_status(state="waking", **payload)
            else:
                self._set_status(state="loading", **payload)
        elif kind == "ready":
            self.loaded = {"asr": True, "llm": bool(self.cfg and self.cfg.polish)}
            self.last_used = time.monotonic()
            self._set_status(state="ready", stage=None, progress=1.0,
                             memory_gb=round(payload.get("memory_gb", 0), 1))
        elif kind == "unloaded":
            what = payload["what"]
            self.loaded = {"asr": what != "all", "llm": False}
            print(f"[otl] unloaded {what}: freed {payload['freed_gb']:.1f} GB, "
                  f"now {payload['memory_gb']:.1f} GB", flush=True)
            self._set_status(state="asleep", stage=what, progress=0.0,
                             memory_gb=round(payload["memory_gb"], 1))
        elif kind == "error":
            msg = f"{type(payload).__name__}: {payload}"
            traceback.print_exception(payload)
            if self.status["state"] != "ready":
                self._set_status(state="error", error=msg)
            else:
                self.hud("error", "出错了：" + msg[:60])
                self._hide_hud_later(2.0)
            self._stop_capture()
        elif kind == "capture_done":
            self._stop_capture()
            eng.send("flush")
        elif kind == "segment":
            self.hud("partial", payload.get("text", ""))
        elif kind == "done":
            self._finish(payload)

    def _finish(self, r):
        audio = r.pop("audio", None)
        text = r["text"]
        if not text:
            self.hud("empty")
            self._hide_hud_later(1.2)
            return
        pasted = False
        if self.store.settings()["auto_paste"]:
            pasted = macos.accessibility_trusted(False)
            AppHelper.callAfter(macos.paste_text, text)  # copies only when not trusted
            if not pasted:
                self.ui("accessibility", False)
        else:
            AppHelper.callAfter(macos.copy_text, text)
        entry = self.store.add({
            "text": text,
            "raw": r["raw"],
            "mode": r.get("mode"),
            "app": self.session.get("app", ""),
            "audio_s": round(r["audio_s"], 2),
            "done_s": round(r["release_to_done_s"], 3),
            "first_s": round(r["release_to_first_char_s"], 3) if r["release_to_first_char_s"] else None,
            "polished": r["polished"],
            "fallback": r["fallback_to_raw"],
            "mic": getattr(self.mic, "name", ""),
        })
        if self.store.settings()["save_audio"] and audio is not None and len(audio):
            self.store.save_audio(entry["id"], audio)
        if pasted:
            self.hud("done", {"text": text, "done_s": r["release_to_done_s"]})
            self._hide_hud_later(0.9)
        else:
            self.hud("copied", {"text": text, "auto": self.store.settings()["auto_paste"]})
            self._hide_hud_later(2.6)
        self.ui("entry", entry)
        self.ui("stats", self.store.stats())

    # ------------------------------------------------------------------ recording

    def _on_audio(self, x):
        if self.engine:
            self.engine.send("audio", x)

    # `recording` is a property so the level ticker can sleep on an Event instead
    # of polling: between dictations it causes no wakeups at all.
    @property
    def recording(self):
        return self._rec_evt.is_set()

    @recording.setter
    def recording(self, on):
        self._rec_evt.set() if on else self._rec_evt.clear()

    def _level_ticker(self):
        while True:
            self._rec_evt.wait()  # blocks (no wakeups) until a dictation starts
            time.sleep(0.04)
            if self.recording and self._js.qsize() < 3:  # drop levels rather than lag
                self._emit(self.hud_win, "hudLevel", self.mic.level)

    def _hotkey_down(self, shift: bool):
        if self.store.settings().get("hotkey_mode") == "toggle" and self.recording:
            self._stop_recording()
            return
        if self.status["state"] in ("asleep", "waking") and self.engine:
            # wake up. If speech recognition is still loaded, start listening right
            # away — the LLM comes back while the user talks (≈3 s).
            if self.status["state"] == "asleep":
                self.engine.send("reload")
                self._set_status(state="waking", stage="llm" if self.loaded["asr"] else "asr", progress=0.0)
            if not self.loaded["asr"]:
                self.hud("waking", self.status)
                self._hide_hud_later(1.6)
                return
        elif self.status["state"] != "ready" or not self.engine:
            self.hud("loading", self.status)
            self._hide_hud_later(1.2)
            return
        self.last_used = time.monotonic()
        if self.recording or self.mic.capturing:
            return
        s = self.store.settings()
        can_translate = models.entry(s["llm_model"]).get("translate", True)
        mode = s["translate_lang"] if shift and can_translate else None
        self.session = {"app": macos.frontmost_app(), "mode": mode}
        self.engine.send("start", mode)
        try:
            self.mic.start()
        except Exception as e:
            self.engine.send("cancel")
            self.hud("error", f"无法打开麦克风：{e}")
            self._hide_hud_later(2.0)
            return
        self.recording = True
        if self.ptt:
            self.ptt.active = True
        toggle = s.get("hotkey_mode") == "toggle"
        self.hud("listening", {"mode": mode, "hint": "再按一次结束，esc 放弃" if toggle else "松开结束，esc 放弃"})

    def _hotkey_up(self):
        if self.store.settings().get("hotkey_mode") == "toggle":
            return  # in toggle mode releasing the key means nothing
        self._stop_recording()

    def _stop_recording(self):
        if not self.recording:
            return
        self.last_used = time.monotonic()
        if self.ptt:
            self.ptt.active = False
        self.recording = False
        self.engine.send("end", time.perf_counter())
        self.hud("thinking", self.session)
        # safety net: if the engine never reports capture_done, stop anyway
        self._capture_timer = threading.Timer(1.5, self._force_flush)
        self._capture_timer.start()

    def _hotkey_cancel(self):
        if self.ptt:
            self.ptt.active = False
        self.recording = False
        self._stop_capture()
        if self.engine:
            self.engine.send("cancel")
        self.hud("hidden")

    def _force_flush(self):
        if self.mic.capturing:
            self._stop_capture()
            if self.engine:
                self.engine.send("flush")

    def _stop_capture(self):
        if self._capture_timer:
            self._capture_timer.cancel()
            self._capture_timer = None
        self.mic.stop()

    def _accessibility_watch(self):
        """Permission is usually granted while we're running. Notice it, then
        rebuild the hotkey tap (a tap can only be created once we're trusted)
        and drop the banner — no restart needed."""
        trusted = macos.accessibility_trusted(False)
        while True:
            time.sleep(1.5)
            now = macos.accessibility_trusted(False)
            if now != trusted:
                trusted = now
                print(f"[otl] accessibility -> {now}", flush=True)
                AppHelper.callAfter(self._install_hotkey)
                self.ui("accessibility", now)

    # ------------------------------------------------------------------ memory

    def _idle_watch(self):
        """Unload models after the configured idle time. Checked every 15 s."""
        while True:
            time.sleep(15)
            s = self.store.settings()
            mins = float(s.get("unload_after_min") or 0)
            if (mins <= 0 or not self.engine or self.recording or self.mic.capturing
                    or self.status["state"] != "ready"):
                continue
            if time.monotonic() - self.last_used >= mins * 60:
                self.unload(s.get("unload_what", "llm"))

    def unload(self, what="llm"):
        if self.engine and not self.recording and self.status["state"] == "ready":
            self.engine.send("unload", what)

    def wake(self):
        if self.engine and self.status["state"] == "asleep":
            self.engine.send("reload")
            self._set_status(state="waking", stage="llm" if self.loaded["asr"] else "asr", progress=0.0)

    # ------------------------------------------------------------------ queries / commands

    def configure_mic(self):
        s = self.store.settings()
        try:
            self.mic.configure(resolve_mic(s["mic"]), bool(s["keep_mic_open"]))
        except Exception:
            traceback.print_exc()

    def catalog(self):
        return models.catalog(self.store.settings(), self.downloads)

    def apply_live_settings(self):
        """Settings that don't need a model reload."""
        if not self.cfg:
            return
        s = self.store.settings()
        self.cfg.fast_path = s["fast_path"]
        self.cfg.post_roll_max_ms = int(s["post_roll_ms"])
        self.cfg.language = None if s["language"] == "auto" else s["language"]

    def set_dictionary(self, words):
        words = self.store.set_dictionary(words)
        if self.cfg and self.engine:
            self.cfg.dictionary[:] = words
            self.engine.send("rebuild_prompts")
        return words


class Api:
    """Methods callable from the web UI as window.pywebview.api.<name>()."""

    def __init__(self, ctl: Controller):
        self._c = ctl

    def state(self):
        c = self._c
        s = c.store.settings()
        status = dict(c.status)
        if status["state"] == "downloading":
            status["progress"] = c.download_progress()
        return {
            "status": status,
            "settings": s,
            "hotkey_label": macos.hotkey_label(s["hotkey"]),
            "hotkey_presets": [{"spec": p, "label": macos.hotkey_label(p)} for p in macos.PRESETS],
            "accessibility": macos.accessibility_trusted(False),
            "stats": c.store.stats(),
        }

    def models(self):
        return self._c.catalog()

    def unload_models(self, what="llm"):
        self._c.unload(what)
        return True

    def wake_models(self):
        self._c.wake()
        return True

    def set_hotkey(self, spec):
        spec = macos.normalize_hotkey(spec)
        self._c.store.update_settings(hotkey=spec)
        AppHelper.callAfter(self._c._install_hotkey)
        return self.state()

    def pause_hotkey(self, paused):
        """the settings page is recording a new hotkey: don't start dictation meanwhile"""
        if self._c.ptt:
            self._c.ptt.paused = bool(paused)
        return True

    def mics(self):
        s = self._c.store.settings()
        return {"mics": list_mics(), "selected": s["mic"], "resolved": resolve_mic(s["mic"])}

    def history(self, limit=50, offset=0, query=""):
        return self._c.store.history(int(limit), int(offset), query or "")

    def delete_entry(self, entry_id):
        self._c.store.delete(entry_id)
        return self._c.store.stats()

    def clear_history(self):
        self._c.store.clear()
        return self._c.store.stats()

    def copy(self, text):
        AppHelper.callAfter(macos.copy_text, text)
        return True

    def dictionary(self):
        return self._c.store.dictionary()

    def save_dictionary(self, words):
        return self._c.set_dictionary(list(words))

    def set_setting(self, key, value):
        c = self._c
        before = c.store.settings()
        s = c.store.update_settings(**{key: value})
        if key == "hotkey":
            AppHelper.callAfter(c._install_hotkey)
        elif key == "translate_lang" and c.cfg and c.engine:
            c.cfg.translate_lang = value or None
            c.engine.send("rebuild_prompts")
        elif key in ("asr_model", "llm_model") and before[key] != value:
            c.load_engine()
        elif key in ("mic", "keep_mic_open"):
            c.configure_mic()
        else:
            c.apply_live_settings()
        return self.state()

    def select_model(self, kind, repo):
        return self.set_setting("asr_model" if kind == "asr" else "llm_model", repo.strip())

    def download(self, repo):
        self._c.downloads.start(repo.strip())
        return True

    def request_accessibility(self):
        trusted = macos.accessibility_trusted(True)
        if not trusted:
            AppHelper.callAfter(macos.open_privacy_pane, "Accessibility")
        return trusted

    def client_log(self, msg):
        """Diagnostics from the web UI (e.g. whether the painting baked)."""
        print(f"[ui] {msg}", flush=True)
        return True
