"""成章 Chengzhang engine: VAD segmentation -> Qwen3-ASR -> LLM polish.

All MLX work (VAD, ASR, LLM) runs on one worker thread: recent MLX streams are
thread-local, so models must be loaded and used on the same thread. Callers talk
to the engine only through `Engine.send()` and the `on_event` callback.
"""
from __future__ import annotations

import copy
import queue
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from . import prompts

SR = 16000
FRAME = 512  # Silero window at 16 kHz = 32 ms
FILLERS = ("嗯", "呃", "啊", "额", "那个", "就是", "然后", "不对", "不是", "对吧")


def looping(text: str, span: int = 8, times: int = 4) -> bool:
    """True when some `span`-char chunk repeats `times`+ times — a decoding loop."""
    t = re.sub(r"\s+", "", text)
    if len(t) < span * times:
        return False
    seen = {}
    for i in range(0, len(t) - span + 1):
        k = t[i : i + span]
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= times:
            return True
    return False


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LN")


@dataclass
class Config:
    asr_model: str = "mlx-community/Qwen3-ASR-0.6B-8bit"
    llm_model: str = "mlx-community/Qwen3.5-4B-MLX-4bit"
    vad_model: str = "mlx-community/silero-vad"
    language: str | None = "Chinese"
    dictionary: list[str] = field(default_factory=list)
    translate: str | None = None  # default mode, e.g. "英文" (CLI --translate)
    translate_lang: str | None = None  # extra mode available per utterance (app: Shift+hotkey)
    polish: bool = True
    fast_path: bool = True
    asr_context: bool = False  # feed previous segments' text to ASR as context
    vad_threshold: float = 0.5
    cut_silence_ms: int = 300
    # once a segment is long, accept shorter pauses so the tail left after release stays short
    soft_cut_after_s: float = 4.0
    soft_cut_silence_ms: int = 150
    min_segment_s: float = 1.5
    max_segment_s: float = 25.0
    max_tokens: int = 1024
    llm_prompt: str = "default"  # "default" = our rules + examples; "refiner" = AgenticASR's own
    compare: bool = False  # also transcribe the whole utterance in one pass
    # Post-roll: people often release the key before the last syllable ends.
    # After release keep capturing until VAD sees this much silence, at most post_roll_max_ms.
    post_roll_max_ms: int = 300
    post_roll_silence_ms: int = 100
    incremental_prefill: bool = True  # prefill LLM with each segment while user speaks


# ----------------------------------------------------------------------------- VAD


class Segmenter:
    """Cuts the live stream at pauses so ASR can run while the user is still speaking."""

    def __init__(self, vad, cfg: Config):
        self.vad, self.cfg = vad, cfg
        self.reset()

    def reset(self):
        self.chunks: list[np.ndarray] = []
        self.total = 0  # samples received
        self.vad_pos = 0  # samples fed to VAD
        self.pending = np.empty(0, np.float32)
        self.state = self.vad.initial_state(sample_rate=SR)
        self.cut = 0  # start of the current (not yet transcribed) segment
        self.first_speech = None  # first speech frame start since last cut
        self.silence_frames = 0

    def audio(self, start=0, end=None) -> np.ndarray:
        if len(self.chunks) > 1:
            self.chunks = [np.concatenate(self.chunks)]
        a = self.chunks[0] if self.chunks else np.empty(0, np.float32)
        return a[start:end]

    def _seg_start(self):
        pad = int(0.1 * SR)
        return max(self.cut, (self.first_speech or self.cut) - pad)

    def push(self, x: np.ndarray) -> list[tuple[int, int]]:
        self.chunks.append(x)
        self.total += len(x)
        self.pending = np.concatenate([self.pending, x])
        segs = []
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            p, self.state = self.vad.feed(frame, self.state, sample_rate=SR)
            p = float(p.item()) if hasattr(p, "item") else float(np.asarray(p).reshape(-1)[0])
            start, self.vad_pos = self.vad_pos, self.vad_pos + FRAME
            if p >= self.cfg.vad_threshold:
                if self.first_speech is None:
                    self.first_speech = start
                self.silence_frames = 0
            else:
                self.silence_frames += 1
            if self.first_speech is None:
                continue
            seg_len = (self.vad_pos - self._seg_start()) / SR
            silence_ms = self.silence_frames * FRAME * 1000 / SR
            need_ms = (
                self.cfg.soft_cut_silence_ms if seg_len >= self.cfg.soft_cut_after_s
                else self.cfg.cut_silence_ms
            )
            if silence_ms >= need_ms and seg_len >= self.cfg.min_segment_s:
                cut_at = self.vad_pos - self.silence_frames * FRAME // 2
            elif seg_len >= self.cfg.max_segment_s:
                cut_at = self.vad_pos
            else:
                continue
            segs.append((self._seg_start(), cut_at))
            self.cut, self.first_speech = cut_at, None
        return segs

    def finish(self) -> tuple[int, int] | None:
        if self.first_speech is None:
            return None
        return (self._seg_start(), self.total)


# ----------------------------------------------------------------------------- ASR


class Transcriber:
    def __init__(self, cfg: Config):
        from mlx_audio.stt import load

        self.cfg = cfg
        self.model = load(cfg.asr_model)

    def __call__(self, wav: np.ndarray, context: str = "") -> str:
        kw = {}
        if self.cfg.dictionary:
            kw["hotwords"] = self.cfg.dictionary
        if context:
            kw["system_prompt"] = context
        return self.model.generate(wav, language=self.cfg.language, **kw).text.strip()


def join_segments(texts: list[str]) -> str:
    out = ""
    for t in texts:
        if out and out[-1].isascii() and out[-1].isalnum() and t[:1].isascii() and t[:1].isalnum():
            out += " "
        out += t
    return out


# ----------------------------------------------------------------------------- LLM


class Polisher:
    """LLM cleanup. One cached prompt prefix per mode: None = polish only,
    "<语言>" = polish + translate. All modes are prefilled at load, so switching
    per utterance costs nothing."""

    def __init__(self, cfg: Config, modes=(None,)):
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self.cfg = cfg
        self.model, self.tok = load(cfg.llm_model)
        self.build(modes)

    def build(self, modes=(None,)):
        """(Re)prefill the cached prompt for each mode. Call again after the
        dictionary or translate language changes — no model reload needed."""
        from mlx_lm.models.cache import make_prompt_cache

        self.modes = {}
        for mode in dict.fromkeys(modes):
            a, b = self._tokens("甲", mode), self._tokens("乙", mode)
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            prefix = a[:n]
            cache = make_prompt_cache(self.model)
            for i in range(0, len(prefix), 512):
                self.model(mx.array(prefix[i : i + 512])[None], cache=cache)
                mx.eval([c.state for c in cache])
            # template tokens that follow the user text (end-of-turn + assistant header)
            k = 0
            while k < min(len(a), len(b)) - len(prefix) and a[-1 - k] == b[-1 - k]:
                k += 1
            self.modes[mode] = {"prefix": prefix, "cache": cache, "tail": a[len(a) - k :]}
        self.live = None
        self.live_mode = None

    # Incremental mode: prefill transcript pieces while the user is still talking,
    # so after release only the last piece + template tail is left to prefill.
    def begin(self, mode=None):
        self.live_mode = mode
        self.live = copy.deepcopy(self.modes[mode]["cache"])

    def feed(self, text: str):
        toks = self.tok.encode(text, add_special_tokens=False)
        if not toks:
            return
        self.model(mx.array(toks)[None], cache=self.live)
        mx.eval([c.state for c in self.live])

    def _tokens(self, text: str, mode=None) -> list[int]:
        if self.cfg.llm_prompt == "refiner":  # fine-tuned on exactly this prompt
            msgs = [{"role": "system", "content": prompts.REFINER_SYSTEM}, {"role": "user", "content": text}]
        else:
            msgs = prompts.messages(text, self.cfg.dictionary, mode)
        return self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True
        )

    def needs_llm(self, raw: str, mode=None) -> bool:
        if mode or not self.cfg.fast_path:
            return True
        return len(norm(raw)) > 12 or any(f in raw for f in FILLERS)

    def __call__(self, raw: str, on_text=None, incremental: bool = False, mode=None) -> dict:
        """Polish `raw`. With incremental=True, `raw` must equal the text passed
        to feed() since begin(mode); only the template tail is prefilled here."""
        from mlx_lm import stream_generate

        t0 = time.perf_counter()
        m = self.modes[mode]
        if incremental and self.live is not None and self.live_mode == mode:
            cache, prompt = self.live, m["tail"]
        else:
            full = self._tokens(raw, mode)
            assert full[: len(m["prefix"])] == m["prefix"], "prompt prefix mismatch"
            cache, prompt = copy.deepcopy(m["cache"]), full[len(m["prefix"]) :]
        self.live = None
        t_copy = time.perf_counter() - t0
        out, ttft, resp = "", None, None
        for resp in stream_generate(
            self.model, self.tok, prompt=prompt,
            max_tokens=self.cfg.max_tokens, prompt_cache=cache,
        ):
            if resp.text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out += resp.text
                if on_text:
                    on_text(resp.text)
        out = out.split("</think>")[-1].strip()
        out = re.sub(r"\s*<KEY>\[.*?\]\s*$", "", out, flags=re.S).strip()  # refiner's entity list
        return {
            "text": out,
            "ttft_s": ttft,
            "total_s": time.perf_counter() - t0,
            "cache_copy_s": t_copy,
            "prompt_tokens_after_release": len(prompt),
            "incremental": incremental,
            "gen_tokens": resp.generation_tokens if resp else 0,
            "gen_tps": resp.generation_tps if resp else 0.0,
        }

    def suspicious(self, raw: str, out: str, mode=None) -> bool:
        """Output that is much shorter/longer than the input probably isn't a cleanup;
        a phrase repeated over and over is a decoding loop."""
        if looping(out):
            return True
        if mode or "翻译" in raw:
            return not out
        r, o = len(norm(raw)), len(norm(out))
        return o == 0 or o < 0.35 * r or o > 1.8 * r + 20


# ----------------------------------------------------------------------------- engine


class Engine(threading.Thread):
    """Commands: ("start",) ("audio", ndarray) ("end", t_release) ("flush",)
    ("unload", "llm"|"all") ("reload",) ("quit",)
    Events via on_event(kind, payload): "ready", "segment", "capture_done", "raw",
    "text", "done", "error".

    After ("end", t) the caller keeps sending audio until "capture_done" (post-roll),
    then may send ("flush",) to force finalization if the stream stopped early."""

    def __init__(self, cfg: Config, on_event):
        super().__init__(daemon=True, name="otl-mlx")
        self.cfg, self.on_event = cfg, on_event
        self.q: queue.Queue = queue.Queue()

    def send(self, *msg):
        self.q.put(msg)

    def run(self):
        try:
            mx.set_default_stream(mx.new_stream(mx.gpu))
            self._load()
        except Exception as e:  # surface load errors to the caller
            self.on_event("error", e)
            return
        while True:
            msg = self.q.get()
            try:
                if msg[0] == "quit":
                    return
                getattr(self, "_on_" + msg[0])(*msg[1:])
            except Exception as e:
                self.on_event("error", e)

    def _load(self):
        from mlx_audio.vad import load as load_vad

        t = time.perf_counter()
        self.on_event("loading", {"stage": "vad", "progress": 0.05})
        self.vad = load_vad(self.cfg.vad_model)
        self.asr = self.polisher = None
        self._load_models(t)

    def _load_models(self, t0, reload=False):
        """Load whatever isn't in memory yet (ASR and/or LLM), then warm up."""
        noise = (np.random.default_rng(0).standard_normal(SR * 2) * 0.01).astype(np.float32)
        if self.asr is None:
            self.on_event("loading", {"stage": "asr", "progress": 0.15, "reload": reload})
            self.asr = Transcriber(self.cfg)
            self.asr(noise)  # compile Metal kernels so the first real utterance is fast
        if self.polisher is None and self.cfg.polish:
            self.on_event("loading", {"stage": "llm", "progress": 0.45, "reload": reload})
            self.polisher = Polisher(self.cfg, modes=self._modes())
            self.on_event("loading", {"stage": "warmup", "progress": 0.85, "reload": reload})
            self.polisher("嗯那个我们明天开会吧")
        if not reload:
            Segmenter(self.vad, self.cfg).push(noise)
            self.seg = Segmenter(self.vad, self.cfg)
        self.on_event("ready", {"load_s": time.perf_counter() - t0, "reload": reload,
                                "memory_gb": mx.get_active_memory() / 1e9})

    def _on_unload(self, what="llm"):
        """Free model weights (and the cached prompt) to give memory back to the
        system. VAD stays: it's tiny. ("reload",) brings them back."""
        if getattr(self, "active", False):
            return  # never in the middle of an utterance
        before = mx.get_active_memory()
        self.polisher = None
        if what == "all":
            self.asr = None
        import gc

        gc.collect()
        mx.clear_cache()
        self.on_event("unloaded", {"what": what, "freed_gb": (before - mx.get_active_memory()) / 1e9,
                                   "memory_gb": mx.get_active_memory() / 1e9})

    def _on_reload(self):
        if self.asr is not None and (self.polisher is not None or not self.cfg.polish):
            self.on_event("ready", {"load_s": 0.0, "reload": True,
                                    "memory_gb": mx.get_active_memory() / 1e9})
            return
        self._load_models(time.perf_counter(), reload=True)

    def _on_start(self, mode="default"):
        if self.asr is None:  # caller should have sent ("reload",) first; be safe
            self._load_models(time.perf_counter(), reload=True)
        self.mode = self.cfg.translate if mode == "default" else mode
        if self.polisher and self.mode not in self.polisher.modes:
            self.mode = None  # e.g. translate requested but the model can't translate
        self.active = True
        self.release_t = None
        self.seg.reset()
        self.texts: list[str] = []
        self.seg_log: list[dict] = []
        if self.polisher and self.cfg.incremental_prefill:
            self.polisher.begin(self.mode)

    def _transcribe(self, start: int, end: int):
        t = time.perf_counter()
        ctx = join_segments(self.texts)[-150:] if self.cfg.asr_context else ""
        text = self.asr(self.seg.audio(start, end), context=ctx)
        dt = time.perf_counter() - t
        t = time.perf_counter()
        if text:
            before = join_segments(self.texts)
            self.texts.append(text)
            if self.polisher and self.polisher.live is not None:
                self.polisher.feed(join_segments(self.texts)[len(before) :])
        info = {"start_s": start / SR, "end_s": end / SR, "asr_s": dt,
                "llm_prefill_s": time.perf_counter() - t, "text": text}
        self.seg_log.append(info)
        self.on_event("segment", info)

    def _on_audio(self, x: np.ndarray):
        if not getattr(self, "active", False):
            return  # audio arriving after finalization
        for s, e in self.seg.push(x.astype(np.float32, copy=False)):
            self._transcribe(s, e)
        if self.release_t is not None:
            self._maybe_finalize()

    def _on_end(self, t_release: float):
        if not getattr(self, "active", False):
            return
        self.release_t = t_release
        self.release_sample = self.seg.total
        self._maybe_finalize()

    def _on_cancel(self):
        self.active = False
        self.release_t = None
        if self.polisher:
            self.polisher.live = None

    def _modes(self):
        if self.cfg.llm_prompt == "refiner":  # it can't translate
            return [None]
        modes = (self.cfg.translate, None, self.cfg.translate_lang)
        return [m for m in modes if m or m is None]

    def _on_rebuild_prompts(self):
        if self.polisher:
            t = time.perf_counter()
            self.polisher.build(self._modes())
            self.on_event("prompts_rebuilt", {"s": time.perf_counter() - t})

    def _on_flush(self):
        if getattr(self, "active", False) and self.release_t is not None:
            self._finalize()

    def _maybe_finalize(self):
        after_ms = (self.seg.total - self.release_sample) * 1000 / SR
        silent_ms = self.seg.silence_frames * FRAME * 1000 / SR
        if (
            self.seg.first_speech is None  # nothing pending since the last cut
            or silent_ms >= self.cfg.post_roll_silence_ms
            or after_ms >= self.cfg.post_roll_max_ms
        ):
            self._finalize()

    def _finalize(self):
        self.active = False
        t_release = self.release_t
        post_roll_s = (self.seg.total - self.release_sample) / SR
        self.on_event("capture_done", post_roll_s)
        tail = self.seg.finish()
        if tail:
            self._transcribe(*tail)
        raw = join_segments(self.texts)
        t_raw = time.perf_counter()
        self.on_event("raw", raw)

        first_text = [None]

        def on_text(chunk):
            if first_text[0] is None:
                first_text[0] = time.perf_counter()
            self.on_event("text", chunk)

        llm = None
        final, fallback = raw, False
        if raw and self.polisher and self.polisher.needs_llm(raw, self.mode):
            llm = self.polisher(raw, on_text, incremental=self.cfg.incremental_prefill,
                                mode=self.mode)
            if self.polisher.suspicious(raw, llm["text"], self.mode):
                fallback = True
            else:
                final = llm["text"]
        elif raw:
            on_text(raw)
        t_done = time.perf_counter()

        res = {
            "raw": raw,
            "text": final,
            "fallback_to_raw": fallback,
            "polished": llm is not None,
            "audio_s": self.seg.total / SR,
            "post_roll_s": post_roll_s,
            "mode": self.mode,
            "segments": self.seg_log,
            "release_to_raw_s": t_raw - t_release,
            "release_to_first_char_s": (first_text[0] - t_release) if first_text[0] else None,
            "release_to_done_s": t_done - t_release,
            "llm": llm,
        }
        if self.cfg.compare and raw:
            res["raw_whole"] = self.asr(self.seg.audio())
        res["audio"] = self.seg.audio()
        self.on_event("done", res)
