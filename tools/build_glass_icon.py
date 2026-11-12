"""Build the 成章 icon as an Icon Composer document (.icon) and render it with
Apple's Liquid Glass renderer (ictool, ships with Xcode 26+).

Layers (front to back):
  disc         sun / moon fill only           -> glass
  reflections  broken light on the water      -> glass
  background   sky, water, glows, stars       -> no glass
The night artwork is the dark-appearance specialization of each layer.

usage: python tools/build_glass_icon.py
  -> design/icon/chengzhang.icon, design/icon/liquid-{light,dark}.png
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import draw_icon as d

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "design/icon/chengzhang.icon"
ICTOOL = "/Applications/Xcode.app/Contents/Applications/Icon Composer.app/Contents/Executables/ictool"


def to_img(rgb, alpha=None):
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if alpha is None:
        img = Image.fromarray(rgb, "RGB")
    else:
        a = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)[..., None]
        img = Image.fromarray(np.concatenate([rgb, a], -1), "RGBA")
    return img.resize((1024, 1024), Image.LANCZOS)


def background_layer(t):
    img = d.background(t)
    img = d.light_path(img, t)
    blend = d.over if t["bloom_mode"] == "over" else d.screen
    # glow around the reflections (their glass bodies live on their own layer)
    bot = d.col(t["refl_bot"])
    for y, h, pieces, k in d.REFLECTION:
        for dx, w in pieces:
            sdf = d.capsule(d.CX + dx, y, w, h)
            ends = 0.20 + 0.80 * np.clip(1 - (np.abs(d.X - (d.CX + dx)) / (w / 2)) ** 2.2, 0, 1)
            img = blend(img, bot, np.clip(1 - sdf / 0.03, 0, 1) ** 2 * 0.40 * k * ends)
    # bloom around the disc
    dd = np.hypot(d.X - d.CX, d.Y - d.CY) / d.R
    bc, ba = t["bloom"]
    img = blend(img, d.col(bc), np.clip(1 - (dd - 1) / 1.7, 0, 1) ** 2.4 * ba)
    img = blend(img, d.col(bc), np.clip(1 - (dd - 1) / 0.40, 0, 1) ** 2 * ba * 0.8)
    return to_img(img)


def reflections_layer(t):
    """Solid shapes (glass needs a crisp silhouette); rows and stroke ends fade by
    blending their colour into the water behind them instead of by transparency."""
    top, bot = d.col(t["refl_top"]), d.col(t["refl_bot"])
    rgb = np.zeros((d.S, d.S, 3), np.float32)
    alpha = np.zeros((d.S, d.S), np.float32)
    for y, h, pieces, k in d.REFLECTION:
        water = d.ramp(np.full((1,), (y - d.H) / (1 - d.H)), t["water"])[0]
        for dx, w in pieces:
            sdf = d.capsule(d.CX + dx, y, w, h)
            a = d.aa(sdf)
            ends = 0.30 + 0.70 * np.clip(1 - (np.abs(d.X - (d.CX + dx)) / (w / 2)) ** 2.2, 0, 1)
            v = np.clip((d.Y - (y - h / 2)) / h, 0, 1)[..., None]
            body = top * (1 - v) + bot * v
            mix = np.clip((0.35 + 0.65 * k) * ends, 0, 1)[..., None]
            body = body * mix + water * (1 - mix)
            m = a > alpha
            rgb[m] = body[m]
            alpha = np.maximum(alpha, a)
    return to_img(rgb, alpha)


def disc_layer(t):
    dd = np.hypot(d.X - d.CX, d.Y - d.CY) / d.R
    ox, oy, spread = t["core"]
    core = np.hypot(d.X - (d.CX + ox * d.R), d.Y - (d.CY + oy * d.R)) / (spread * d.R)
    fill = d.ramp(core, t["disc"])
    if t["moon"]:
        fill = d.moon_surface(fill, dd)
    return to_img(fill, d.aa(d.circle(d.CX, d.CY, d.R)))


def layer(name, glass):
    return {
        "name": name,
        # ictool only honours specializations when the list also carries the default,
        # and there is no separate "image-name" key
        "image-name-specializations": [{"value": f"{name}-day.png"},
                                       {"appearance": "dark", "value": f"{name}-night.png"}],
        "glass": glass,
    }


def build():
    if DOC.exists():
        shutil.rmtree(DOC)
    (DOC / "Assets").mkdir(parents=True)
    for tag, t in (("day", d.DAY), ("night", d.NIGHT)):
        background_layer(t).save(DOC / "Assets" / f"background-{tag}.png")
        reflections_layer(t).save(DOC / "Assets" / f"reflections-{tag}.png")
        disc_layer(t).save(DOC / "Assets" / f"disc-{tag}.png")
    doc = {
        "fill-specializations": [{"value": {"solid": "extended-srgb:0.62,0.70,0.78,1.0"}},
                                 {"appearance": "dark", "value": {"solid": "extended-srgb:0.09,0.13,0.24,1.0"}}],
        "groups": [
            {"name": "Disc", "layers": [layer("disc", True)],
             "shadow": {"kind": "layer-color", "opacity": 0.5},
             "translucency": {"enabled": True, "value": 0.25},
             "specular": True},
            {"name": "Reflections", "layers": [layer("reflections", True)],
             "shadow": {"kind": "layer-color", "opacity": 0.12},
             "translucency": {"enabled": True, "value": 0.45},
             "specular": True},
            {"name": "Background", "layers": [layer("background", False)],
             "shadow": {"kind": "none", "opacity": 0},
             "specular": False},
        ],
        "supported-platforms": {"squares": ["macOS"]},
    }
    (DOC / "icon.json").write_text(json.dumps(doc, indent=2))


def render(rendition, out):
    r = subprocess.run([ICTOOL, str(DOC), "--export-image", "--output-file", str(out), "--platform", "macOS",
                        "--rendition", rendition, "--width", "1024", "--height", "1024", "--scale", "1"],
                       capture_output=True, text=True)
    if r.returncode or not out.exists():
        print(rendition, "FAILED:", (r.stdout + r.stderr)[-1500:])
    else:
        print(rendition, "->", out.name)


if __name__ == "__main__":
    build()
    render("Default", ROOT / "design/icon/liquid-light.png")
    render("Dark", ROOT / "design/icon/liquid-dark.png")
