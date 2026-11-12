"""成章 Chengzhang CLI prototype.

  otl                         # mic: Enter to start, Enter to stop
  otl --simulate a.wav b.wav  # feed files at real-time speed (each file = one utterance)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .engine import SR, Config, Engine, norm

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class App:
    def __init__(self, args):
        self.args = args
        dictionary = []
        if args.dict and Path(args.dict).exists():
            dictionary = [
                l.strip() for l in Path(args.dict).read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")
            ]
        self.cfg = Config(
            asr_model=args.asr_model,
            llm_model=args.llm_model,
            language=None if args.language == "auto" else args.language,
            dictionary=dictionary,
            translate=args.translate,
            polish=not args.no_polish,
            fast_path=not args.no_fast_path,
            asr_context=args.asr_context,
            cut_silence_ms=args.cut_silence_ms,
            compare=args.compare,
            post_roll_max_ms=args.post_roll_ms,
            incremental_prefill=not args.no_incremental,
        )
        self.ready = threading.Event()
        self.done = threading.Event()
        self.capture_done = threading.Event()
        self.error = None
        self.results = []
        self.engine = Engine(self.cfg, self.on_event)

    # -------------------------------------------------------------- events

    def on_event(self, kind, payload):
        if kind == "ready":
            print(f"{DIM}模型已加载 {payload['load_s']:.1f}s，预热 {payload['warmup_s']:.1f}s{RESET}")
            self.ready.set()
        elif kind == "capture_done":
            self.capture_done.set()
        elif kind == "segment" and self.args.verbose:
            p = payload
            print(f"{DIM}  [段 {p['start_s']:.1f}–{p['end_s']:.1f}s, ASR {p['asr_s']*1000:.0f}ms] {p['text']}{RESET}")
        elif kind == "raw":
            if self.args.verbose:
                print(f"{DIM}  原始: {payload}{RESET}")
            if not self.args.json:
                sys.stdout.write(BOLD)
        elif kind == "text" and not self.args.json:
            sys.stdout.write(payload)
            sys.stdout.flush()
        elif kind == "done":
            self.finish(payload)
            self.done.set()
        elif kind == "error":
            self.error = payload
            import traceback

            traceback.print_exception(payload)
            self.ready.set()
            self.done.set()

    def finish(self, r):
        audio = r.pop("audio")
        if r["fallback_to_raw"] and not self.args.json:
            print(f"\n{DIM}[润色结果可疑，已回退到原始转写]{RESET}\n{BOLD}{r['text']}{RESET}", end="")
        if "raw_whole" in r:
            a, b = norm(r["raw"]), norm(r["raw_whole"])
            r["seg_vs_whole_cer"] = edit_distance(a, b) / max(len(b), 1)
        if self.args.json:
            print(json.dumps(r, ensure_ascii=False))
        else:
            print(RESET)
            llm = r["llm"]
            parts = [
                f"语音 {r['audio_s']:.1f}s / {len(r['segments'])} 段",
                f"收尾录音 {r['post_roll_s']*1000:.0f}ms",
                f"ASR 收尾 {r['release_to_raw_s']*1000:.0f}ms",
            ]
            if llm:
                parts.append(f"LLM 首字 {llm['ttft_s']*1000:.0f}ms · {llm['gen_tokens']} tok @ {llm['gen_tps']:.0f} tok/s")
            else:
                parts.append("跳过 LLM")
            if r["release_to_first_char_s"] is not None:
                parts.append(f"松开→首字 {r['release_to_first_char_s']*1000:.0f}ms")
            parts.append(f"松开→完成 {r['release_to_done_s']*1000:.0f}ms")
            print(DIM + " · ".join(parts) + RESET)
            if "raw_whole" in r:
                print(f"{DIM}  整段识别: {r['raw_whole']}\n  分段 vs 整段 差异 {r['seg_vs_whole_cer']*100:.1f}%{RESET}")
        if self.args.save and r["raw"]:
            d = Path(self.args.save)
            d.mkdir(parents=True, exist_ok=True)
            stem = d / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            import soundfile as sf

            sf.write(f"{stem}.wav", audio, SR)
            Path(f"{stem}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        if not self.args.no_copy and not self.args.simulate and r["text"]:
            subprocess.run(["pbcopy"], input=r["text"].encode())
        self.results.append(r)

    # -------------------------------------------------------------- modes

    def start(self):
        self.engine.start()
        self.ready.wait()
        if self.error:
            sys.exit(1)

    def utterance(self, feed):
        """feed(send, release) pushes audio, calls release() when the user lets go,
        and keeps pushing audio until self.capture_done is set (post-roll)."""
        self.done.clear()
        self.capture_done.clear()
        self.engine.send("start")
        feed(lambda x: self.engine.send("audio", x),
             lambda: self.engine.send("end", time.perf_counter()))
        self.engine.send("flush")  # stream closed: finalize with whatever arrived
        self.done.wait()

    def run_simulate(self):
        import soundfile as sf

        for path in self.args.simulate:
            wav, sr = sf.read(path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            assert sr == SR, f"{path}: need 16 kHz, got {sr}"
            if not self.args.json:
                print(f"\n{DIM}▶ {path} ({len(wav)/SR:.1f}s){RESET}")

            def feed(send, release, wav=wav):
                # release N ms before the file ends (user lets go mid-word), then keep
                # "recording": the rest of the file, then mic silence, until capture_done
                release_at = max(0, len(wav) - int(SR * self.args.release_early_ms / 1000))
                silence = np.zeros(int(SR * 1.0), np.float32)
                stream = np.concatenate([wav, silence])
                t0, released = time.perf_counter(), False
                for i in range(0, len(stream), 512):
                    if not released and i >= release_at:
                        release()
                        released = True
                    if released and self.capture_done.is_set():
                        return
                    send(stream[i : i + 512])
                    if self.args.speed > 0:  # pace like a live microphone
                        due = t0 + (i + 512) / SR / self.args.speed
                        time.sleep(max(0.0, due - time.perf_counter()))
                    elif released:
                        self.capture_done.wait(0.002)  # let the engine react between chunks

            self.utterance(feed)

    def run_mic(self):
        import sounddevice as sd

        dev = sd.query_devices(kind="input")
        rate = SR
        try:
            sd.check_input_settings(samplerate=SR, channels=1, dtype="float32")
        except Exception:
            rate = int(dev["default_samplerate"])
        print(f"{DIM}麦克风: {dev['name']} @ {rate} Hz{RESET}")
        if rate != SR:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(SR, rate)
            up, down = SR // g, rate // g
        print("按 Enter 开始说话，再按 Enter 结束。Ctrl-C 退出。")
        while True:
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                return

            def feed(send, release):
                def cb(indata, frames, t, status):
                    x = indata[:, 0].copy()
                    if rate != SR:
                        x = resample_poly(x, up, down).astype(np.float32)
                    send(x)

                with sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                                    blocksize=int(rate * 0.032), callback=cb):
                    print(f"{DIM}● 录音中… 按 Enter 结束{RESET}", flush=True)
                    try:
                        input()
                    except (EOFError, KeyboardInterrupt):
                        pass
                    release()
                    # keep the mic open for the post-roll (engine decides, ≤ post_roll_ms)
                    self.capture_done.wait(timeout=1.0)

            self.utterance(feed)
            print()


def main(argv=None):
    p = argparse.ArgumentParser(prog="otl", description="成章 Chengzhang 命令行原型")
    p.add_argument("--simulate", nargs="+", metavar="WAV", help="用 16 kHz wav 文件代替麦克风（按实时速度输入）")
    p.add_argument("--speed", type=float, default=1.0, help="模拟输入速度倍数，0 = 不限速")
    p.add_argument("--asr-model", default=Config.asr_model)
    p.add_argument("--llm-model", default=Config.llm_model)
    p.add_argument("--language", default="Chinese", help="ASR 语言，auto = 自动检测")
    p.add_argument("--dict", default="dictionary.txt", help="个人词典，每行一个词")
    p.add_argument("--translate", metavar="LANG", help="整理后翻译成该语言，例如 英文")
    p.add_argument("--no-polish", action="store_true", help="只做 ASR")
    p.add_argument("--no-fast-path", action="store_true", help="短句也走 LLM")
    p.add_argument("--asr-context", action="store_true", help="把前面几段的文字作为 ASR 上下文")
    p.add_argument("--cut-silence-ms", type=int, default=Config.cut_silence_ms, help="停顿多久切一段")
    p.add_argument("--no-incremental", action="store_true", help="不在说话时预填充 LLM（对比用）")
    p.add_argument("--post-roll-ms", type=int, default=300,
                   help="松开后最多再录多久（遇到静音会提前结束），0 = 立即停止")
    p.add_argument("--release-early-ms", type=int, default=0,
                   help="模拟模式：在文件结束前多少毫秒\"松开\"（模拟最后一个字还没说完）")
    p.add_argument("--compare", action="store_true", help="另外整段识别一次，比较分段识别的差异")
    p.add_argument("--save", metavar="DIR", help="保存每次录音和结果（用来建评测集）")
    p.add_argument("--no-copy", action="store_true", help="不复制结果到剪贴板")
    p.add_argument("--json", action="store_true", help="每次输出一行 JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="显示每段识别结果")
    args = p.parse_args(argv)

    app = App(args)
    app.start()
    if args.simulate:
        app.run_simulate()
    else:
        app.run_mic()
    app.engine.send("quit")


if __name__ == "__main__":
    main()
