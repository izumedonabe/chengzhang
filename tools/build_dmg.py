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


def slim(site: Path):
    """drop what an end user never needs: tests, caches, dev-only packages"""
    for name in ("PyObjCTest", "pyarrow", "pip", "setuptools"):
        for p in site.glob(name):
            shutil.rmtree(p, ignore_errors=True)
    for pattern in ("**/__pycache__", "numpy/**/tests", "scipy/**/tests", "PIL/**/tests"):
        for p in site.glob(pattern):
            shutil.rmtree(p, ignore_errors=True)


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
    slim(site)
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
    run("codesign", "--force", "--sign", "-", APP)

    # DMG: the app next to a shortcut to /Applications
    stage = BUILD / "dmg"
    stage.mkdir()
    shutil.move(str(APP), stage / APP.name)
    (stage / "Applications").symlink_to("/Applications")
    (ROOT / "dist").mkdir(exist_ok=True)
    dmg = ROOT / "dist" / f"Chengzhang-{version}-arm64.dmg"
    dmg.unlink(missing_ok=True)
    run("hdiutil", "create", "-volname", "成章", "-srcfolder", stage, "-ov", "-format", "UDZO",
        "-imagekey", "zlib-level=9", dmg)
    shutil.move(str(stage / APP.name), APP)  # keep build/成章.app around for testing
    print(f"\nbuilt {dmg}  ({dmg.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
