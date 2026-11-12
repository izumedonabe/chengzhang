"""Build dist/成章.app — a real macOS app bundle around the dev venv.

The bundle's executable is a tiny C launcher (packaging/launcher.c) that hosts
Python in-process, so the app shows up as 成章 everywhere (Dock, menu bar,
Cmd-Tab, privacy settings) instead of python3.12.

usage: python tools/build_app.py      -> dist/成章.app
"""
import plistlib
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dist" / "成章.app"
VENV_PY = ROOT / ".venv" / "bin" / "python"
EXE = "Chengzhang"


def main():
    libdir = sysconfig.get_config_var("LIBDIR")
    include = sysconfig.get_config_var("INCLUDEPY")
    ver = sysconfig.get_config_var("VERSION")  # e.g. 3.12
    if not (Path(libdir) / f"libpython{ver}.dylib").exists():
        sys.exit(f"need a shared libpython{ver}.dylib in {libdir}")

    if APP.exists():
        shutil.rmtree(APP)
    macos = APP / "Contents" / "MacOS"
    res = APP / "Contents" / "Resources"
    macos.mkdir(parents=True)
    res.mkdir(parents=True)

    subprocess.run([
        "clang", "-O2", "-o", str(macos / EXE), str(ROOT / "packaging/launcher.c"),
        f"-I{include}", f"-L{libdir}", f"-lpython{ver}", f"-Wl,-rpath,{libdir}",
        f'-DVENV_PYTHON="{VENV_PY}"', "-mmacosx-version-min=14.0",
    ], check=True)

    shutil.copy(ROOT / "chengzhang/app/assets/chengzhang.icns", res / "chengzhang.icns")
    info = {
        "CFBundleName": "成章",
        "CFBundleDisplayName": "成章",
        "CFBundleExecutable": EXE,
        "CFBundleIdentifier": "app.chengzhang.dictation",
        "CFBundleIconFile": "chengzhang",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "0.0.1",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "14.0",
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": "成章需要使用麦克风，才能把你说的话写成文字。声音只在这台 Mac 上处理。",
    }
    with open(APP / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(info, f)

    # ad-hoc signature with an identifier-based designated requirement. Plain ad-hoc
    # signing pins permissions (Accessibility, Microphone) to this exact binary's hash,
    # so every rebuild silently orphans them; pinning to the bundle id keeps them.
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-",
                    "--requirements", '=designated => identifier "app.chengzhang.dictation"', str(APP)], check=True)
    print("built", APP)


if __name__ == "__main__":
    main()
