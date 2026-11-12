"""Build the app icons (.icns + 1024 PNG) from the Liquid Glass renders.

ictool renders the squircle edge-to-edge; macOS app icons keep the body at
824/1024 with a soft shadow underneath, so we place it on that grid here.

usage: python tools/build_icon.py      (after tools/build_glass_icon.py)
  -> chengzhang/app/assets/chengzhang{,-dark}.{icns,png}
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "design/icon"
OUT = ROOT / "chengzhang/app/assets"


def place(render: Image.Image, canvas=1024, body=824) -> Image.Image:
    art = render.convert("RGBA").resize((body, body), Image.LANCZOS)
    off = (canvas - body) // 2
    icon = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    shadow.paste(Image.new("RGBA", (body, body), (0, 0, 0, 80)), (off, off + 12), art.getchannel("A"))
    icon.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
    icon.alpha_composite(art, (off, off))
    return icon


def icns(icon: Image.Image, path: Path):
    tmp = Path(tempfile.mkdtemp()) / "icon.iconset"
    tmp.mkdir()
    for s in (16, 32, 128, 256, 512):
        icon.resize((s, s), Image.LANCZOS).save(tmp / f"icon_{s}x{s}.png")
        icon.resize((s * 2, s * 2), Image.LANCZOS).save(tmp / f"icon_{s}x{s}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", str(tmp), "-o", str(path)], check=True)
    shutil.rmtree(tmp.parent)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for render, name in (("liquid-light.png", "chengzhang"), ("liquid-dark.png", "chengzhang-dark")):
        icon = place(Image.open(SRC / render))
        icon.save(OUT / f"{name}.png")
        icns(icon, OUT / f"{name}.icns")
        print("wrote", OUT / f"{name}.icns")


if __name__ == "__main__":
    main()
