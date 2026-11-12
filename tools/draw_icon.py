"""Draw the 成章 app icon — day (sun) and night (moon) — as luminous glass.

Everything is analytic (signed-distance shapes + gradients) so edges stay crisp
and glows stay smooth. Units are fractions of the artwork side S.

usage: python tools/draw_icon.py      -> design/icon/glass-{day,night}.png
"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "design/icon"
S = 1648                       # 2x the 824 px icon body; downsampled later
H = 0.60                       # horizon
CX, CY, R = 0.50, 0.355, 0.150  # the disc
LIGHT = np.array([-0.62, -0.78])  # light comes from the upper left

Y, X = (np.mgrid[0:S, 0:S].astype(np.float32) + 0.5) / S


# ----------------------------------------------------------------------------- primitives

def col(c):
    return np.array(c, np.float32) / 255


def ramp(t, stops):
    """t: array in 0..1 -> rgb (…, 3) by piecewise-linear stops [(pos, (r,g,b)), …]"""
    t = np.clip(t, 0, 1)
    out = np.zeros(t.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, [p for p, _ in stops], [v[c] / 255 for _, v in stops])
    return out


def aa(sdf):
    return np.clip(0.5 - sdf * S, 0, 1)


def circle(cx, cy, r):
    return np.hypot(X - cx, Y - cy) - r


def capsule(cx, cy, w, h):
    a = max(w / 2 - h / 2, 0)
    dx = np.maximum(np.abs(X - cx) - a, 0)
    return np.hypot(dx, Y - cy) - h / 2


def over(img, color, alpha):
    a = np.clip(alpha, 0, 1)[..., None]
    c = color if np.ndim(color) == 3 else np.broadcast_to(color, img.shape)
    return img * (1 - a) + c * a


def screen(img, color, alpha):
    """light adds up: screen blend, so glows brighten without going muddy"""
    a = np.clip(alpha, 0, 1)[..., None]
    c = color if np.ndim(color) == 3 else np.broadcast_to(color, img.shape)
    return 1 - (1 - img) * (1 - c * a)


def gauss(v, s):
    return np.exp(-(v / s) ** 2)


# ----------------------------------------------------------------------------- themes

DAY = dict(
    sky=[(0.0, (112, 142, 178)), (0.45, (166, 184, 204)), (0.82, (226, 204, 188)), (1.0, (247, 214, 186))],
    water=[(0.0, (214, 196, 186)), (0.12, (170, 172, 180)), (1.0, (78, 102, 128))],
    sky_glow=((250, 176, 136), 0.30),
    disc=[(0.0, (255, 160, 104)), (0.55, (246, 112, 60)), (1.0, (230, 86, 44))],
    core=(0.0, -0.95, 2.1),   # light falls from the top: centre offset (in radii) and spread
    bloom_mode="over",
    bloom=((246, 120, 66), 0.42),
    rim=(255, 246, 232),
    caustic=((255, 214, 170), 0.30),
    path=((255, 168, 104), 0.42),
    refl_top=(255, 168, 108), refl_bot=(238, 100, 50),
    horizon=((255, 232, 206), 0.95),
    stars=[],
    moon=False,
)

NIGHT = dict(
    sky=[(0.0, (12, 18, 40)), (0.5, (26, 38, 74)), (0.85, (48, 64, 104)), (1.0, (66, 82, 122))],
    water=[(0.0, (60, 76, 114)), (0.12, (34, 46, 80)), (1.0, (8, 12, 26))],
    sky_glow=((150, 176, 230), 0.42),
    disc=[(0.0, (255, 252, 242)), (0.6, (244, 238, 222)), (1.0, (214, 206, 188))],
    core=(-0.12, -0.18, 1.12),
    bloom_mode="screen",
    bloom=((176, 198, 246), 0.55),
    rim=(255, 255, 255),
    caustic=((226, 232, 255), 0.35),
    path=((178, 200, 246), 0.30),
    refl_top=(236, 242, 255), refl_bot=(140, 164, 214),
    horizon=((168, 190, 236), 0.75),
    stars=[(0.205, 0.165, 1.0), (0.80, 0.115, 0.65), (0.700, 0.250, 0.42), (0.13, 0.36, 0.35)],
    moon=True,
)

# broken light on the water: rows of (y, height, [(x offset, width), …], strength)
REFLECTION = [
    (0.652, 0.034, [(-0.010, 0.36)], 1.00),
    (0.705, 0.028, [(-0.092, 0.14), (0.076, 0.15)], 0.85),
    (0.755, 0.024, [(0.020, 0.18)], 0.70),
    (0.800, 0.020, [(-0.046, 0.07), (0.060, 0.05)], 0.55),
    (0.840, 0.017, [(0.008, 0.05)], 0.42),
]

# moon surface: maria (soft dark seas) and craters (dx, dy, radius) in disc radii
MARIA = [(-0.28, -0.18, 0.36, 0.26, 0.22), (0.24, 0.08, 0.30, 0.40, 0.18), (-0.06, 0.42, 0.26, 0.16, 0.14)]
CRATERS = [(0.36, -0.34, 0.13), (-0.40, 0.30, 0.10), (0.06, 0.10, 0.075), (-0.16, -0.52, 0.065),
           (0.52, 0.36, 0.085), (-0.58, -0.08, 0.055), (0.20, 0.58, 0.05)]


# ----------------------------------------------------------------------------- painting

def background(t):
    sky = ramp(Y / H, t["sky"])
    water = ramp((Y - H) / (1 - H), t["water"])
    blend = np.clip((Y - (H - 0.006)) / 0.012, 0, 1)[..., None]
    img = sky * (1 - blend) + water * blend
    # the air around the disc is lit
    d = np.hypot(X - CX, (Y - CY) * 1.05)
    gc, ga = t["sky_glow"]
    img = screen(img, col(gc), np.clip(1 - d / 0.62, 0, 1) ** 2.2 * ga * (Y < H + 0.02))
    # the horizon catches the light near the disc
    hc, ha = t["horizon"]
    img = screen(img, col(hc), gauss(Y - H, 0.0042) * gauss(X - CX, 0.20) * ha)
    img = screen(img, col(hc), gauss(Y - H, 0.0016) * 0.35)
    # a few faint ripples across the water
    for i, yy in enumerate((0.69, 0.745, 0.80, 0.86, 0.915)):
        img = screen(img, col(hc), gauss(Y - yy, 0.0018) * gauss(X - CX, 0.30) * 0.05 * (1 - i * 0.12))
    for x, y, b in t["stars"]:
        img = star(img, x, y, b)
    return img


def star(img, x0, y0, b):
    d = np.hypot(X - x0, Y - y0)
    c = col((246, 244, 236))
    img = screen(img, c, np.clip(1 - d / (0.0055 * (0.6 + b * 0.6)), 0, 1) ** 1.5 * b)
    img = screen(img, c, np.clip(1 - d / 0.03, 0, 1) ** 3 * 0.35 * b)
    if b > 0.8:  # only the brightest twinkles
        rays = (gauss(X - x0, 0.0013) * np.clip(1 - np.abs(Y - y0) / 0.034, 0, 1) ** 2
                + gauss(Y - y0, 0.0013) * np.clip(1 - np.abs(X - x0) / 0.034, 0, 1) ** 2)
        img = screen(img, c, rays * 0.75 * b)
    return img


def light_path(img, t):
    """the glow the disc lays on the water, under the broken reflections"""
    pc, pa = t["path"]
    below = np.clip((Y - H) / (1 - H), 0, 1)
    a = gauss(X - CX, 0.10 + 0.06 * below) * (Y > H) * (1 - below) ** 1.3 * pa
    return screen(img, col(pc), a)


def reflections(img, t):
    top, bot = col(t["refl_top"]), col(t["refl_bot"])
    for y, h, pieces, k in REFLECTION:
        for dx, w in pieces:
            sdf = capsule(CX + dx, y, w, h)
            ends = 0.20 + 0.80 * np.clip(1 - (np.abs(X - (CX + dx)) / (w / 2)) ** 2.2, 0, 1)
            # soft halo, then the glass body, then its bright top edge
            halo = over if t["bloom_mode"] == "over" else screen
            img = halo(img, bot, np.clip(1 - sdf / 0.026, 0, 1) ** 2 * (sdf > 0) * 0.45 * k * ends)
            v = np.clip((Y - (y - h / 2)) / h, 0, 1)[..., None]
            body = top * (1 - v) + bot * v
            img = over(img, body, aa(sdf) * 0.88 * k * ends)
            edge = aa(sdf) * gauss(Y - (y - h * 0.26), h * 0.13) * 0.55 * k * ends
            img = screen(img, col((255, 250, 240)), edge)
    return img


def disc(img, t):
    sdf = circle(CX, CY, R)
    d = np.hypot(X - CX, Y - CY) / R
    body = aa(sdf)
    # bloom: two layers of light around the disc
    bc, ba = t["bloom"]
    blend = over if t["bloom_mode"] == "over" else screen  # on a pale sky, additive light turns white
    img = blend(img, col(bc), np.clip(1 - (d - 1) / 1.7, 0, 1) ** 2.4 * (d > 0.98) * ba)
    img = blend(img, col(bc), np.clip(1 - (d - 1) / 0.40, 0, 1) ** 2 * (d > 0.98) * ba * 0.8)
    # luminous fill: brightest a little above centre, as if glowing from within
    ox, oy, spread = t["core"]
    core = np.hypot(X - (CX + ox * R), Y - (CY + oy * R)) / (spread * R)
    fill = ramp(core, t["disc"])
    if t["moon"]:
        fill = moon_surface(fill, d)
    img = over(img, fill, body)
    # glass: bright rim strongest toward the light, a softer one opposite
    nx, ny = (X - CX) / (d * R + 1e-6), (Y - CY) / (d * R + 1e-6)
    lit = nx * LIGHT[0] + ny * LIGHT[1]         # +1 where the surface normal points at the light
    facing = np.clip(lit, 0, 1)
    back = np.clip(-lit, 0, 1)
    ring = gauss(d - 0.965, 0.02) * body
    img = screen(img, col(t["rim"]), ring * (0.08 + 0.92 * facing ** 2) * 0.55)
    cc, ca = t["caustic"]
    img = screen(img, col(cc), gauss(d - 0.86, 0.06) * back ** 2 * body * ca)  # light refracted through
    # a small, soft specular glint
    g = np.hypot((X - (CX - 0.42 * R)) / 1.4, Y - (CY - 0.52 * R)) / (0.16 * R)
    img = screen(img, col((255, 255, 255)), np.clip(1 - g, 0, 1) ** 2 * 0.0 * body)  # glint off: reads as plastic
    return img


def moon_surface(fill, d):
    dark = col((168, 162, 150))
    out = fill.copy()
    for mx, my, rx, ry, a in MARIA:
        e = ((X - (CX + mx * R)) / (rx * R)) ** 2 + ((Y - (CY + my * R)) / (ry * R)) ** 2
        out = over(out, dark, np.clip(1 - e, 0, 1) ** 1.2 * a)
    for cx, cy, cr in CRATERS:
        px, py, rr = CX + cx * R, CY + cy * R, cr * R
        dd = np.hypot(X - px, Y - py)
        inside = np.clip((rr - dd) * S / 1.5, 0, 1)
        ux, uy = (X - px) / (dd + 1e-6), (Y - py) / (dd + 1e-6)
        toward = ux * LIGHT[0] + uy * LIGHT[1]          # +1 on the side facing the light
        wall = gauss(dd / rr - 0.78, 0.2) * inside
        out = over(out, dark, inside * 0.30)                                    # floor
        out = over(out, col((120, 116, 108)), wall * np.clip(toward, 0, 1) * 0.55)  # shadowed inner wall
        out = over(out, col((255, 252, 244)), wall * np.clip(-toward, 0, 1) * 0.65)  # lit inner wall
        lip = gauss(dd / rr - 1.06, 0.07)
        out = over(out, col((255, 253, 246)), lip * np.clip(toward, 0, 1) * 0.5)    # sunlit outer lip
    # gentle limb darkening toward the lower right
    shade = np.clip(d, 0, 1) ** 3 * np.clip(-(X - CX) * LIGHT[0] / R - (Y - CY) * LIGHT[1] / R, 0, 1) * 0.22
    return over(out, col((150, 146, 140)), shade)


def plate(img):
    """the glass slab itself: a diagonal sheen and a bright edge, strongest at the top"""
    diag = (X * 0.55 + Y) / 1.55
    img = screen(img, col((255, 255, 255)), np.clip(1 - diag / 0.46, 0, 1) ** 2.4 * 0.16)
    n = 5.0
    se = np.abs(2 * X - 1) ** n + np.abs(2 * Y - 1) ** n
    edge = gauss(se - 0.985, 0.03) * (se < 1.02)
    img = screen(img, col((255, 255, 255)), edge * (0.12 + 0.5 * np.clip(1 - Y * 1.7, 0, 1)))
    return img


def paint(t):
    img = background(t)
    img = light_path(img, t)
    img = reflections(img, t)
    img = disc(img, t)
    img = plate(img)
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in (("day", DAY), ("night", NIGHT)):
        paint(t).save(OUT / f"glass-{name}.png")
        print("drew", name)


if __name__ == "__main__":
    main()
