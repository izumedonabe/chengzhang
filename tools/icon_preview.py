"""Preview an icon candidate the way macOS shows it: squircle mask, Apple's
1024 grid (824 px artwork with ~100 px margin), at Dock/Finder/menu sizes.

usage: python tools/icon_preview.py SOURCE OUT.png [--crop x0 y0 size]
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def squircle_mask(size: int, n: float = 5.0) -> Image.Image:
    """superellipse |x|^n + |y|^n = 1, close to Apple's continuous-corner shape"""
    scale = 4
    S = size * scale
    m = Image.new("L", (S, S), 0)
    px = m.load()
    r = S / 2
    for y in range(S):
        dy = abs((y + 0.5 - r) / r) ** n
        if dy >= 1:
            continue
        dx = (1 - dy) ** (1 / n) * r
        x0, x1 = int(r - dx), int(r + dx)
        for x in range(max(0, x0), min(S, x1)):
            px[x, y] = 255
    return m.resize((size, size), Image.LANCZOS)


def make_icon(art: Image.Image, canvas: int = 1024, body: int = 824) -> Image.Image:
    art = art.convert("RGB").resize((body, body), Image.LANCZOS)
    mask = squircle_mask(body)
    icon = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    off = (canvas - body) // 2
    # soft drop shadow like the system's
    shadow = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    sh = Image.new("RGBA", (body, body), (0, 0, 0, 90))
    shadow.paste(sh, (off, off + 12), mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    icon.alpha_composite(shadow)
    body_img = Image.new("RGBA", (body, body))
    body_img.paste(art, (0, 0))
    body_img.putalpha(mask)
    icon.alpha_composite(body_img, (off, off))
    return icon


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    im = Image.open(src).convert("RGB")
    if "--crop" in sys.argv:
        i = sys.argv.index("--crop")
        x0, y0, s = (int(v) for v in sys.argv[i + 1 : i + 4])
        im = im.crop((x0, y0, x0 + s, y0 + s))
    icon = make_icon(im)
    icon.save(out.with_name(out.stem + "-1024.png"))

    # contact sheet: Dock-ish 256 / 128, Finder 64, menu/list 32, and 16
    sizes = [256, 128, 64, 32, 16]
    pad = 28
    W = sum(sizes) + pad * (len(sizes) + 1)
    H = max(sizes) + pad * 2 + 30
    for bg, tag in (((236, 239, 237), "light"), ((38, 40, 44), "dark")):
        sheet = Image.new("RGB", (W, H), bg)
        x = pad
        d = ImageDraw.Draw(sheet)
        for s in sizes:
            ic = icon.resize((s, s), Image.LANCZOS)
            sheet.paste(ic, (x, pad + (max(sizes) - s)), ic)
            d.text((x, H - 26), f"{s}px", fill=(120, 130, 135))
            x += s + pad
        sheet.save(out.with_name(f"{out.stem}-sizes-{tag}.png"))
    print("ok", out.parent)


if __name__ == "__main__":
    main()
