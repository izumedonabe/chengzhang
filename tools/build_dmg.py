"""Build a self-contained 成章.app and wrap it in a DMG.

The app carries its own Python (a relocatable python-build-standalone CPython
installed by `uv python install 3.12`) with every dependency inside
Contents/Resources/python. Speech and language models are not bundled; the app
downloads the ones you pick on first launch.

usage: python tools/build_dmg.py
  -> build/成章.app, dist/Chengzhang-<version>-arm64.dmg
"""
import os
import plistlib
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
APP = BUILD / "成章.app"
PY_VERSION = "3.12"
EXE = "Chengzhang"


def run(*cmd, **kw):
    print("+", " ".join(str(c) for c in cmd)[:160], flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def standalone_python() -> Path:
    """a relocatable python-build-standalone CPython from uv's own store — never a
    conda or system Python, which can't be copied into a bundle"""
    store = Path(subprocess.run(["uv", "python", "dir"], capture_output=True, text=True, check=True).stdout.strip())
    found = sorted(store.glob(f"cpython-{PY_VERSION}.*-macos-aarch64-none"))
    if not found:
        run("uv", "python", "install", PY_VERSION)
        found = sorted(store.glob(f"cpython-{PY_VERSION}.*-macos-aarch64-none"))
    py = found[-1]
    assert (py / "lib" / f"libpython{PY_VERSION}.dylib").exists(), py
    return py


# Packages pulled in as dependencies but never imported by the app at runtime
# (traced with a full dictation + polish + translate run, the window, and the
# model-download path). Keep anything loaded lazily on a real path: cffi
# (sounddevice opens the mic through it), hf_xet (fast model downloads),
# protobuf/google (slow-tokenizer fallback), miniaudio (audio file decoding).
UNUSED_PACKAGES = (
    "scipy", "scipy.libs",                        # 53 MB: replaced by chengzhang/resample.py
    "pydantic", "pydantic_core",                  # only for CLIs / servers we don't ship
    "pygments", "rich", "typer", "click", "markdown_it", "mdurl", "shellingham",
    "PyObjCTest", "pyarrow", "pip", "setuptools",
)
# transformers ships code for hundreds of architectures; tokenizers/configs only
# ever touch these (plus families a user may pick in the model list)
TRANSFORMERS_KEEP = {
    "auto", "encoder_decoder", "whisper", "audioflamingo3",
    "llama", "qwen2", "qwen2_audio", "qwen2_moe", "qwen3", "qwen3_moe", "qwen3_next",
    "qwen3_5", "qwen3_5_moe", "qwen3_asr", "qwen4_exp",
    "gemma", "gemma2", "gemma3", "gemma4", "gemma4_unified", "minicpm3",
}
# mlx-audio is a toolkit (TTS, music, speech-to-speech, ~40 ASR models); we use
# Qwen3-ASR and Silero VAD
MLX_AUDIO_DROP = ("tts", "sts", "music", "codec", "lid", "tests", "server.py", "server_inference.py", "convert.py")
STT_KEEP = {"qwen3_asr", "qwen2_audio", "__init__.py", "base.py"}
VAD_KEEP = {"silero_vad", "__init__.py"}
STDLIB_DROP = ("idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3", "pydoc_data", "test", "unittest/test")


def _rm(p: Path):
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p, ignore_errors=True)
    elif p.exists() or p.is_symlink():
        p.unlink()


def slim(site: Path, pyhome: Path):
    """drop what an end user never needs: unused packages, other models' code,
    tests, headers, caches"""
    for name in UNUSED_PACKAGES:
        for p in site.glob(name):
            _rm(p)
        for p in site.glob(f"{name}-*.dist-info"):
            _rm(p)
    tf = site / "transformers" / "models"
    for p in tf.iterdir():
        if p.is_dir() and p.name not in TRANSFORMERS_KEEP:
            _rm(p)
    ma = site / "mlx_audio"
    for name in MLX_AUDIO_DROP:
        _rm(ma / name)
    for p in (ma / "stt" / "models").iterdir():
        if p.name not in STT_KEEP:
            _rm(p)
    for p in (ma / "vad" / "models").iterdir():
        if p.name not in VAD_KEEP:
            _rm(p)
    _rm(site / "mlx" / "include")  # C++ headers for building extensions
    _rm(site / "mlx" / "share")    # CMake files
    stdlib = pyhome / "lib" / f"python{PY_VERSION}"
    for name in STDLIB_DROP:
        _rm(stdlib / name)
    for p in stdlib.glob("config-*"):  # static lib + Makefile for building extensions
        for f in p.glob("*.a"):
            _rm(f)
    for pattern in ("**/__pycache__", "**/tests", "**/test", "numpy/**/tests", "PIL/**/tests"):
        for p in site.glob(pattern):
            _rm(p)


def main():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    src = standalone_python()
    print("python:", src)

    shutil.rmtree(BUILD, ignore_errors=True)
    contents = APP / "Contents"
    res = contents / "Resources"
    (contents / "MacOS").mkdir(parents=True)
    res.mkdir(parents=True)
    pyhome = res / "python"
    shutil.copytree(src, pyhome, symlinks=True)
    (pyhome / "lib" / f"python{PY_VERSION}" / "EXTERNALLY-MANAGED").unlink(missing_ok=True)

    # the app and all its dependencies, installed into the bundled interpreter
    bundled_py = pyhome / "bin" / f"python{PY_VERSION}"
    run("uv", "pip", "install", "--python", bundled_py, "--compile-bytecode", ROOT)
    site = pyhome / "lib" / f"python{PY_VERSION}" / "site-packages"
    slim(site, pyhome)
    for p in (pyhome / "bin").iterdir():  # console scripts point at this build machine
        if p.name not in (f"python{PY_VERSION}", "python3", "python"):
            p.unlink()

    # launcher, linked to the bundled libpython by a path relative to itself
    lib = pyhome / "lib" / f"libpython{PY_VERSION}.dylib"
    exe = contents / "MacOS" / EXE
    run("clang", "-O2", "-o", exe, ROOT / "packaging/launcher_bundle.c",
        f"-I{pyhome}/include/python{PY_VERSION}", f"-L{pyhome}/lib", f"-lpython{PY_VERSION}",
        "-mmacosx-version-min=14.0")
    old = subprocess.run(["otool", "-D", str(lib)], capture_output=True, text=True).stdout.split("\n")[1].strip()
    run("install_name_tool", "-change", old, f"@executable_path/../Resources/python/lib/libpython{PY_VERSION}.dylib", exe)
    run("install_name_tool", "-id", f"@rpath/libpython{PY_VERSION}.dylib", lib)
    _rm(pyhome / "include")  # C headers: only needed to compile the launcher above

    shutil.copy(ROOT / "chengzhang/app/assets/chengzhang.icns", res / "chengzhang.icns")
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump({
            "CFBundleName": "成章",
            "CFBundleDisplayName": "成章",
            "CFBundleExecutable": EXE,
            "CFBundleIdentifier": "app.chengzhang.dictation",
            "CFBundleIconFile": "chengzhang",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": version,
            "CFBundleVersion": version,
            "LSMinimumSystemVersion": "14.0",
            "LSArchitecturePriority": ["arm64"],
            "LSApplicationCategoryType": "public.app-category.productivity",
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": "成章需要使用麦克风，才能把你说的话写成文字。声音只在这台 Mac 上处理。",
        }, f)

    # ad-hoc signature (no Apple Developer ID): every Mach-O inside, then the bundle
    for root, _, files in os.walk(pyhome):
        for name in files:
            p = Path(root) / name
            if p.suffix in (".so", ".dylib") or (p.parent.name == "bin" and os.access(p, os.X_OK) and not p.is_symlink()):
                subprocess.run(["codesign", "--force", "--sign", "-", str(p)], capture_output=True)
    # identifier-based requirement: permissions survive updates (see build_app.py)
    run("codesign", "--force", "--sign", "-", "--requirements", '=designated => identifier "app.chengzhang.dictation"', APP)

    # DMG: the app next to a shortcut to /Applications
    stage = BUILD / "dmg"
    stage.mkdir()
    shutil.move(str(APP), stage / APP.name)
    (stage / "Applications").symlink_to("/Applications")
    (ROOT / "dist").mkdir(exist_ok=True)
    dmg = ROOT / "dist" / f"Chengzhang-{version}-arm64.dmg"
    dmg.unlink(missing_ok=True)
    run("hdiutil", "create", "-volname", "成章", "-srcfolder", stage, "-ov", "-format", "ULMO", dmg)
    shutil.move(str(stage / APP.name), APP)  # keep build/成章.app around for testing
    print(f"\nbuilt {dmg}  ({dmg.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
