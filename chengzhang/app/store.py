"""Local persistence: settings, dictionary, history. All under
~/Library/Application Support/Chengzhang."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from ..engine import norm

APP_DIR = Path.home() / "Library" / "Application Support" / "Chengzhang"
OLD_APP_DIR = Path.home() / "Library" / "Application Support" / "OpenTypeLess"  # before the rename
REPO_DICT = Path(__file__).resolve().parents[2] / "dictionary.txt"

DEFAULT_SETTINGS = {
    "asr_model": "mlx-community/Qwen3-ASR-0.6B-8bit",
    "llm_model": "mlx-community/Qwen3.5-4B-MLX-4bit",  # "" = no polishing
    "hotkey": {"kind": "modifier", "keycode": 61},  # see macos.normalize_hotkey
    "hotkey_mode": "hold",  # hold: talk while held · toggle: press to start, press again to stop
    "translate_lang": "英文",
    "language": "Chinese",  # ASR language; "auto" = detect
    "post_roll_ms": 300,
    "fast_path": True,
    "auto_paste": True,
    "appearance": "auto",  # auto | light | dark
    "mic": "auto",  # "auto" = built-in mic first; or a device name
    "keep_mic_open": False,
    "save_audio": False,
    # free model memory after this many idle minutes: 0 = never.
    # "unload_what": "llm" keeps speech recognition loaded (first words still instant)
    "unload_after_min": 0,
    "unload_what": "llm",
}

TYPING_CPM = 50  # rough Chinese typing speed, for "time saved"


class Store:
    def __init__(self, root: Path = APP_DIR):
        if root == APP_DIR and not root.exists() and OLD_APP_DIR.exists():
            OLD_APP_DIR.rename(root)  # keep history, dictionary and settings across the rename
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.settings_path = root / "settings.json"
        self.dict_path = root / "dictionary.txt"
        self.history_path = root / "history.jsonl"
        if not self.dict_path.exists():
            seed = REPO_DICT.read_text(encoding="utf-8") if REPO_DICT.exists() else ""
            self.dict_path.write_text(seed, encoding="utf-8")

    # ---------------------------------------------------------------- settings

    def settings(self) -> dict:
        s = dict(DEFAULT_SETTINGS)
        if self.settings_path.exists():
            try:
                s.update(json.loads(self.settings_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        return s

    def update_settings(self, **kw) -> dict:
        with self._lock:
            s = self.settings()
            s.update({k: v for k, v in kw.items() if k in DEFAULT_SETTINGS})
            self.settings_path.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
            return s

    # ---------------------------------------------------------------- dictionary

    def dictionary(self) -> list[str]:
        lines = self.dict_path.read_text(encoding="utf-8").splitlines()
        return [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]

    def set_dictionary(self, words: list[str]) -> list[str]:
        clean = list(dict.fromkeys(w.strip() for w in words if w.strip()))
        self.dict_path.write_text(
            "# 个人词典：每行一个词，同时用于语音识别和润色。\n" + "\n".join(clean) + "\n",
            encoding="utf-8",
        )
        return clean

    # ---------------------------------------------------------------- history

    def add(self, entry: dict) -> dict:
        entry = {"id": uuid.uuid4().hex[:12], "ts": time.time(), **entry}
        with self._lock, self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def _all(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        out = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def history(self, limit=50, offset=0, query="") -> list[dict]:
        items = [e for e in reversed(self._all()) if not e.get("deleted")]
        if query:
            q = query.lower()
            items = [e for e in items if q in e.get("text", "").lower() or q in e.get("raw", "").lower()]
        return items[offset : offset + limit]

    def delete(self, entry_id: str):
        with self._lock:
            items = [e for e in self._all() if e.get("id") != entry_id]
            self.history_path.write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in items), encoding="utf-8"
            )

    def clear(self):
        with self._lock:
            self.history_path.write_text("", encoding="utf-8")

    def save_audio(self, entry_id: str, audio) -> Path:
        import soundfile as sf

        d = self.root / "recordings"
        d.mkdir(exist_ok=True)
        path = d / f"{entry_id}.wav"
        sf.write(path, audio, 16000)
        return path

    def stats(self) -> dict:
        items = self._all()
        today0 = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        week0 = today0 - 6 * 86400

        def agg(es):
            # count what was said (the raw transcript), so translations don't count letters
            chars = sum(len(norm(e.get("raw") or e.get("text", ""))) for e in es)
            audio = sum(e.get("audio_s", 0) for e in es)
            done = [e["done_s"] for e in es if e.get("done_s") is not None]
            return {
                "count": len(es),
                "chars": chars,
                "audio_min": audio / 60,
                "cpm": chars / (audio / 60) if audio > 1 else 0,
                "avg_done_s": sum(done) / len(done) if done else None,
                "saved_min": max(0.0, chars / TYPING_CPM - audio / 60),
            }

        return {
            "today": agg([e for e in items if e.get("ts", 0) >= today0]),
            "week": agg([e for e in items if e.get("ts", 0) >= week0]),
            "all": agg(items),
        }
