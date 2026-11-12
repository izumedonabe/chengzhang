"""Model catalog shown in the app, plus download state/progress.
Sizes are download sizes; memory is a rough runtime figure. Users can also
enter any Hugging Face repo id."""
from __future__ import annotations

import os
import threading
from pathlib import Path

ASR = [
    {
        "id": "mlx-community/Qwen3-ASR-0.6B-8bit",
        "name": "Qwen3-ASR 0.6B",
        "blurb": "反应快，日常口述足够准。",
        "facts": {"大小": "1.0 GB", "内存": "约 2 GB"},
        "size_gb": 1.01,
        "recommended": True,
    },
    {
        "id": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "name": "Qwen3-ASR 1.7B",
        "blurb": "更准一些，口音重、环境吵、专业词多时更稳；稍慢。",
        "facts": {"大小": "2.5 GB", "内存": "约 4 GB"},
        "size_gb": 2.47,
    },
]

LLM = [
    {
        "id": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "name": "Qwen3.5 4B",
        "blurb": "整理得最干净：去口头禅、理解改口、自动分条。",
        "facts": {"大小": "3.1 GB", "内存": "约 3.5 GB"},
        "size_gb": 3.06,
        "recommended": True,
    },
    {
        "id": "mlx-community/Qwen3.5-2B-MLX-4bit",
        "name": "Qwen3.5 2B",
        "blurb": "更快、更省内存，口头禅偶尔删不干净。",
        "facts": {"大小": "1.8 GB", "内存": "约 2 GB"},
        "size_gb": 1.75,
    },
    {
        "id": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "name": "Qwen3 4B Instruct",
        "blurb": "改动更保守，更多保留你的原话。",
        "facts": {"大小": "2.3 GB", "内存": "约 3 GB"},
        "size_gb": 2.28,
    },
    {
        "id": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "name": "Qwen3.5 9B",
        "blurb": "理解力更强，适合长段口述；生成更慢，内存要求更高。",
        "facts": {"大小": "6.0 GB", "内存": "约 7 GB"},
        "size_gb": 5.98,
    },
    {
        # "repo:subfolder" — the repo's root holds full-precision weights; we only
        # need the MLX 4-bit copy in mlx-int4/
        "id": "Andrew0425/AgenticASR-Refiner:mlx-int4",
        "name": "AgenticASR Refiner 1B",
        "blurb": "专为口语转书面语微调的小模型：很快，忠实于原话，不会擅自回复或翻译。"
                 "不修同音错字，长段落和长串数字容易丢内容或写错；不支持翻译。",
        "facts": {"大小": "0.7 GB", "内存": "约 1 GB"},
        "size_gb": 0.68,
        "prompt": "refiner",
        "translate": False,
    },
    {
        "id": "",
        "name": "不整理",
        "blurb": "只做语音识别，原样输出，会保留口头禅和改口。",
        "facts": {},
        "size_gb": 0,
    },
]


LOCAL_MODELS = Path(__file__).resolve().parents[2] / "models"  # project-local copies


def split_spec(spec: str) -> tuple[str, str | None]:
    """'owner/name:subfolder' -> ('owner/name', 'subfolder')"""
    repo, _, sub = spec.partition(":")
    return repo, (sub or None)


def _cache_dir(spec: str) -> Path:
    from huggingface_hub import constants

    repo, _ = split_spec(spec)
    return Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def _local_copy(spec: str) -> Path | None:
    repo, sub = split_spec(spec)
    p = LOCAL_MODELS / repo.split("/")[-1].lower()
    p = p / sub if sub else p
    return p if (p / "config.json").exists() else None


def resolve_path(spec: str) -> str:
    """What mlx_lm.load() should get: a project-local copy, the cached
    subfolder, or the plain repo id."""
    if not spec:
        return spec
    local = _local_copy(spec)
    if local:
        return str(local)
    repo, sub = split_spec(spec)
    if sub:
        from huggingface_hub import snapshot_download

        return str(Path(snapshot_download(repo, allow_patterns=[f"{sub}/*"], local_files_only=True)) / sub)
    return repo


def entry(spec: str) -> dict:
    for m in ASR + LLM:
        if m["id"] == spec:
            return m
    return {}


def is_downloaded(spec: str) -> bool:
    if not spec or _local_copy(spec):
        return True
    from huggingface_hub import snapshot_download

    repo, sub = split_spec(spec)
    try:
        path = snapshot_download(repo, local_files_only=True,
                                 allow_patterns=[f"{sub}/*"] if sub else None)
        return not sub or (Path(path) / sub / "config.json").exists()
    except Exception:
        return False


def _dir_bytes(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class Downloads:
    """Background snapshot_download with progress = bytes on disk / expected bytes."""

    def __init__(self, on_change):
        self.on_change = on_change
        self.active: dict[str, dict] = {}

    def state(self, repo: str) -> dict | None:
        d = self.active.get(repo)
        if not d:
            return None
        if d["status"] == "downloading" and d["expected"]:
            d["progress"] = min(0.99, _dir_bytes(_cache_dir(repo)) / d["expected"])
        return {k: d[k] for k in ("status", "progress", "error")}

    def start(self, repo: str):
        if not repo or (repo in self.active and self.active[repo]["status"] == "downloading"):
            return
        d = self.active[repo] = {"status": "downloading", "progress": 0.0, "error": None, "expected": 0}

        def run():
            from huggingface_hub import HfApi, snapshot_download

            try:
                name, sub = split_spec(repo)
                info = HfApi().model_info(name, files_metadata=True)
                files = [s for s in info.siblings if not sub or s.rfilename.startswith(sub + "/")]
                d["expected"] = sum((s.size or 0) for s in files) or 0
                snapshot_download(name, allow_patterns=[f"{sub}/*"] if sub else None)
                d.update(status="done", progress=1.0)
            except Exception as e:  # network, bad repo id, disk full…
                d.update(status="error", error=str(e)[:200])
            self.on_change(repo)

        threading.Thread(target=run, daemon=True, name=f"dl-{repo}").start()
        self.on_change(repo)


def catalog(settings: dict, downloads: Downloads) -> dict:
    def decorate(items, current):
        out = []
        ids = [m["id"] for m in items]
        if current not in ids:  # a custom repo the user entered
            items = items + [{"id": current, "name": current.split("/")[-1], "blurb": "自定义模型",
                              "facts": {}, "size_gb": None, "custom": True}]
        for m in items:
            out.append({**m, "selected": m["id"] == current,
                        "downloaded": is_downloaded(m["id"]),
                        "download": downloads.state(m["id"])})
        return out

    return {"asr": decorate(ASR, settings["asr_model"]), "llm": decorate(LLM, settings["llm_model"])}
