"""Procedural cartoon remake: a cartoon bear cub rides a bike down a sunny
grass slope toward a cliff-top castle city with a viaduct train and the sea.

Renders an SDR (BT.709) master and pipes it to ffmpeg. Grade it to HLG with
../grade_hlg.sh afterwards. Only numpy + Pillow are needed.

    python3 render.py out_sdr.mp4 [--fps 60] [--seconds 7] [--preview N]
"""
import argparse
import math
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- helpers
def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


def vgrad(w, h, stops):
    """Vertical RGB gradient. stops: list of (pos 0..1, (r,g,b))."""
    ys = np.linspace(0, 1, h)
    out = np.zeros((h, w, 3), np.float32)
    pos = [p for p, _ in stops]
    for ch in range(3):
        col = np.interp(ys, pos, [c[ch] for _, c in stops])
        out[:, :, ch] = col[:, None]
    return out


def value_noise(h, w, cell, seed):
    """Smooth 2-D value noise in [0,1], tileable horizontally."""
    r = np.random.default_rng(seed)
    gh, gw = h // cell + 2, w // cell + 1
    g = r.random((gh, gw)).astype(np.float32)
    g = np.concatenate([g, g[:, :1]], 1)
    y = np.arange(h) / cell
    x = np.arange(w) / cell
    y0 = y.astype(int); x0 = x.astype(int)
    fy = y - y0; fx = x - x0
    fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
    a = g[y0][:, x0]; b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]; d = g[y0 + 1][:, x0 + 1]
    top = a + (b - a) * fx[None]
    bot = c + (d - c) * fx[None]
    return top + (bot - top) * fy[:, None]


def fbm(h, w, cell, seed, octaves=4):
    tot = np.zeros((h, w), np.float32); amp = 1.0; norm = 0
    for o in range(octaves):
        c = max(2, cell >> o)
        tot += amp * value_noise(h, w, c, seed + o); norm += amp; amp *= 0.5
    return tot / norm


def rgba(arr_rgb, alpha):
    a = np.dstack([np.clip(arr_rgb, 0, 255), np.clip(alpha, 0, 255)]).astype(np.uint8)
    return Image.fromarray(a, 'RGBA')


def ss_layer(w, h, scale=2):
    img = Image.new('RGBA', (w * scale, h * scale), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img), scale


def down(img, scale):
    return img.resize((img.width // scale, img.height // scale), Image.LANCZOS)


# ---------------------------------------------------------------- static layers
def make_sky(lw):
    sky = vgrad(lw, H, [(0, (16, 64, 186)), (0.28, (34, 108, 222)),
                        (0.42, (92, 162, 236)), (0.55, (130, 190, 244)), (1, (130, 190, 244))])
    return Image.fromarray(sky.astype(np.uint8)).convert('RGBA')


def make_clouds(lw):
    """Big cumulus banks. Pure-white sunlit tops are the main HDR highlights."""
    img, d, s = ss_layer(lw, 520)
    r = np.random.default_rng(11)
    banks = [(330, 250, 380), (900, 200, 300), (1560, 230, 340), (2300, 200, 300)]
    for bx, by, bw in banks:
        puffs = []
        for _ in range(26):
            px = bx + r.uniform(-bw / 2, bw / 2)
            rad = r.uniform(28, 70) * (1 - abs(px - bx) / bw * 0.9)
            py = by - r.uniform(0, 1) ** 1.5 * (150 - abs(px - bx) * 0.35) + rad * 0.3
            puffs.append((px, py, max(rad, 30)))
        # shadow pass (bluish grey underside), then lit pass offset upward
        for px, py, rad in puffs:
            d.ellipse([(px - rad) * s, (py - rad) * s, (px + rad) * s, (py + rad) * s], fill=(172, 196, 228, 255))
        for px, py, rad in puffs:
            rr = rad * 0.86
            d.ellipse([(px - rr - 6) * s, (py - rr - 14) * s, (px + rr - 6) * s, (py + rr - 14) * s],
                      fill=(255, 255, 255, 255))
    img = down(img, s)
    # soft cartoon edge
    return img.filter(ImageFilter.GaussianBlur(1.2))


def make_far(lw):
    """Sea, distant white chalk cliffs and green headland (parallax ~0.15)."""
    img, d, s = ss_layer(lw, H)
    hz = 420
    # sea
    sea = vgrad(lw * s, (H - hz) * s, [(0, (64, 140, 214)), (0.25, (30, 104, 196)), (1, (14, 64, 150))])
    img.paste(Image.fromarray(sea.astype(np.uint8)).convert('RGBA'), (0, hz * s))
    # sun glints on sea
    r = np.random.default_rng(3)
    for _ in range(260):
        x = r.uniform(0, lw); y = r.uniform(hz + 8, hz + 200)
        L = r.uniform(6, 26) * (1 + (y - hz) / 120)
        d.line([(x * s, y * s), ((x + L) * s, y * s)], fill=(235, 248, 255, 255), width=int(2 * s))
    # distant white cliffs coast receding to the left horizon
    pts = []
    for i in range(0, lw + 1, 20):
        t = i / lw
        top = hz - 10 - 160 * (t ** 1.6) - 18 * math.sin(i * 0.013) - 10 * math.sin(i * 0.051)
        pts.append((i, top))
    bot = [(x, hz + 40 + 140 * (x / lw) ** 1.4) for x, _ in pts]
    poly = pts + bot[::-1]
    d.polygon([(x * s, y * s) for x, y in poly], fill=(222, 226, 236, 255))
    # cliff face shading stripes
    for i in range(0, lw, 14):
        t = i / lw
        top = hz - 10 - 160 * (t ** 1.6) - 18 * math.sin(i * 0.013) - 10 * math.sin(i * 0.051)
        b = hz + 40 + 140 * t ** 1.4
        if (i // 14) % 3 == 0:
            d.line([(i * s, (top + 22) * s), ((i + 6) * s, b * s)], fill=(196, 204, 222, 255), width=int(3 * s))
    # green caps on cliffs
    cap = [(x, y) for x, y in pts] + [(x, y + 22 + 6 * math.sin(x * 0.03)) for x, y in pts[::-1]]
    d.polygon([(x * s, y * s) for x, y in cap], fill=(66, 140, 64, 255))
    # foam line
    d.line([(x * s, (y + 4) * s) for x, y in bot], fill=(255, 255, 255, 255), width=int(5 * s))
    # green headland lower left
    hl = [(0, 640)] + [(x, 640 - 170 * math.exp(-((x - 300) / 420) ** 2) + 12 * math.sin(x * 0.02))
                       for x in range(0, lw + 1, 20)] + [(lw, 640), (lw, H), (0, H)]
    d.polygon([(x * s, y * s) for x, y in hl], fill=(38, 96, 52, 255))
    img = down(img, s)
    # tree speckle on headland
    a = np.array(img).astype(np.float32)
    n = fbm(H, lw, 16, 21)
    mask = (a[:, :, 1] > 80) & (a[:, :, 0] < 50) & (a[:, :, 2] < 70)
    a[mask, :3] *= (0.65 + 0.7 * n[mask])[:, None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA')


def make_city(lw):
    """Giant cliff city with gothic spires, glowing windows and a stone viaduct (parallax ~0.45)."""
    lh = H
    img, d, s = ss_layer(lw, lh)
    r = np.random.default_rng(5)
    x0 = 1150  # city cliff starts here
    # rock wall
    wall = [(x0, 260), (x0 + 260, 170), (x0 + 700, 120), (lw, 100), (lw, 820), (x0 - 40, 820)]
    d.polygon([(x * s, y * s) for x, y in wall], fill=(96, 84, 98, 255))
    cap = [(x0 - 10, 262), (x0 + 260, 168), (x0 + 700, 118), (lw, 98), (lw, 140), (x0 + 700, 160), (x0 + 260, 205), (x0 - 10, 290)]
    d.polygon([(x * s, y * s) for x, y in cap], fill=(84, 150, 62, 255))
    # buildings carved in the wall: towers
    for k in range(42):
        bx = x0 + 20 + k * 52 + r.uniform(-10, 10)
        bw = r.uniform(30, 60)
        bt = r.uniform(170, 430) + max(0, 200 - (bx - x0) * 0.3)
        col = mix((112, 104, 140), (150, 132, 150), r.random())
        d.rectangle([bx * s, bt * s, (bx + bw) * s, 820 * s], fill=col + (255,))
        # spire
        d.polygon([(bx * s, bt * s), ((bx + bw / 2) * s, (bt - r.uniform(40, 110)) * s), ((bx + bw) * s, bt * s)],
                  fill=mix(col, (220, 214, 230), 0.35) + (255,))
        # sunlit left edge
        d.rectangle([bx * s, bt * s, (bx + 5) * s, 820 * s], fill=(214, 206, 222, 255))
        # windows: warm glow
        for wy in np.arange(bt + 18, 760, r.uniform(26, 40)):
            for wx in np.arange(bx + 9, bx + bw - 8, 14):
                if r.random() < 0.55:
                    glow = mix((255, 150, 40), (255, 214, 120), r.random())
                    d.rounded_rectangle([wx * s, wy * s, (wx + 7) * s, (wy + 12) * s], radius=3 * s, fill=glow + (255,))
    # white tower cluster near the viaduct (bright, like the reference's pale spires)
    for k, (tx, tw, tt) in enumerate([(x0 - 110, 40, 330), (x0 - 62, 30, 390), (x0 - 20, 36, 300)]):
        d.rectangle([tx * s, tt * s, (tx + tw) * s, 820 * s], fill=(212, 212, 226, 255))
        d.polygon([(tx * s, tt * s), ((tx + tw / 2) * s, (tt - 90) * s), ((tx + tw) * s, tt * s)], fill=(226, 226, 236, 255))
        d.rectangle([(tx + tw - 10) * s, tt * s, (tx + tw) * s, 820 * s], fill=(176, 180, 206, 255))
    # viaduct: deck curve from lower-left (x0-60, 640) sweeping right to (lw, 470)
    def deck(x):
        t = (x - (x0 - 60)) / (lw - (x0 - 60))
        return 560 - 120 * t + 70 * (1 - t) ** 2
    xs = np.arange(x0 - 60, lw + 1, 4)
    top = [(x, deck(x)) for x in xs]
    body = top + [(x, deck(x) + 34) for x in xs[::-1]]
    d.polygon([(x * s, y * s) for x, y in body], fill=(222, 214, 196, 255))
    d.line([(x * s, (y + 34) * s) for x, y in top], fill=(150, 140, 128, 255), width=int(4 * s))
    # arches
    span = 70
    for ax in np.arange(x0 - 40, lw, span):
        y = deck(ax) + 34
        pw = 14
        d.rectangle([ax * s, y * s, (ax + pw) * s, 840 * s], fill=(214, 204, 186, 255))
        d.rectangle([(ax + pw - 4) * s, y * s, (ax + pw) * s, 840 * s], fill=(160, 150, 136, 255))
        d.arc([(ax + pw) * s, (y - 10) * s, (ax + span) * s, (y + 90) * s], 180, 360, fill=(214, 204, 186, 255), width=int(9 * s))
    img = down(img, s)
    a = np.array(img).astype(np.float32)
    n = fbm(lh, lw, 24, 41)
    rock = (a[:, :, 3] > 0)
    a[rock, :3] *= (0.85 + 0.3 * n[rock])[:, None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA'), deck


def make_trees(lw):
    """Dark green forest band at the foot of the city (parallax ~0.7)."""
    img, d, s = ss_layer(lw, H)
    r = np.random.default_rng(9)
    for _ in range(420):
        x = r.uniform(0, lw); y = r.uniform(700, 860) + 40 * math.sin(x * 0.004)
        rad = r.uniform(18, 46)
        c = mix((20, 70, 36), (60, 128, 54), r.random())
        d.ellipse([(x - rad) * s, (y - rad) * s, (x + rad) * s, (y + rad) * s], fill=c + (255,))
        d.ellipse([(x - rad * 0.6) * s, (y - rad * 0.95) * s, (x + rad * 0.2) * s, (y - rad * 0.3) * s],
                  fill=mix(c, (150, 210, 90), 0.45) + (255,))
    d.rectangle([0, 840 * s, lw * s, H * s], fill=(24, 70, 36, 255))
    return down(img, s)


# ---------------------------------------------------------------- ground plane
GRASS_TEX_H, GRASS_TEX_W = 1024, 1024
_gn = fbm(GRASS_TEX_H, GRASS_TEX_W, 64, 101, 5)
_gs = value_noise(GRASS_TEX_H, GRASS_TEX_W, 4, 202)  # fine blade noise


def _bilerp(tex, fv, fu):
    h, w = tex.shape
    v0 = np.floor(fv); u0 = np.floor(fu)
    tv = (fv - v0).astype(np.float32); tu = (fu - u0).astype(np.float32)
    v0 = v0.astype(np.int64) % h; u0 = u0.astype(np.int64) % w
    v1 = (v0 + 1) % h; u1 = (u0 + 1) % w
    a = tex[v0, u0] + (tex[v0, u1] - tex[v0, u0]) * tu
    b = tex[v1, u0] + (tex[v1, u1] - tex[v1, u0]) * tu
    return a + (b - a) * tv


def grass_tex(u, v):
    """u: across (world x), v: along path (world z). Stretched along v => motion streaks."""
    return _bilerp(_gs, v * 6, u * 40), _bilerp(_gn, v * 1.2, u * 6)


def render_ground(frame, t, cam_z, bob, crest, path_x0, path_x1):
    """Perspective ground plane under a wavy crest line. frame: float32 HxWx3."""
    f = 900.0
    horizon = 520 + bob
    y0 = int(max(crest.min() - 2, horizon + 6))
    ys = np.arange(y0, H)[:, None].astype(np.float32)
    xs = np.arange(W)[None, :].astype(np.float32)
    dy = np.maximum(ys - horizon, 1)
    z = 1.6 * f / dy                       # depth along path
    u = (xs - W * 0.52) * z / f            # lateral world coord
    v = z + cam_z
    blade, patch = grass_tex(u, v)
    # path: a dirt track curving down-slope
    q = np.clip((H - ys) / (H - crest.mean()), 0, 1)          # 0 at bottom, 1 at crest
    pcx = path_x0 + (path_x1 - path_x0) * q ** 0.8 + 60 * np.sin(q * 3.0 + t * 0.4) * q
    pd = np.abs(xs - pcx) * z / f                              # lateral world distance
    path = np.clip((0.42 - pd) / 0.10, 0, 1)
    rut = np.clip(1 - np.abs(pd - 0.2) / 0.05, 0, 1) * path
    # colour: sunlit yellow-green, brighter toward crest (backlit rim), darker near camera
    near = np.clip((ys - y0) / (H - y0), 0, 1)
    g_lit = np.array([150, 196, 56], np.float32)
    g_dark = np.array([36, 88, 24], np.float32)
    shade = 0.25 + 0.75 * (0.55 * blade + 0.45 * patch)
    base = g_dark + (g_lit - g_dark) * (shade * (1 - 0.65 * near))[..., None]
    dirt = np.array([170, 150, 98], np.float32) * (0.8 + 0.3 * blade)[..., None]
    col = base * (1 - path[..., None] * 0.85) + dirt * (path[..., None] * 0.85)
    col *= (1 - 0.25 * rut)[..., None]
    # rim light at crest (big HDR highlight band like the reference)
    dist_crest = ys - crest[None, :]
    rim = np.exp(-np.maximum(dist_crest, 0) / 22.0)
    col += (np.array([255, 250, 200], np.float32) - col) * (0.75 * rim)[..., None]
    # haze toward crest (far) for depth
    far = np.clip(1 - dy / 260, 0, 1)
    col = col * (1 - 0.25 * far[..., None]) + np.array([220, 236, 200], np.float32) * (0.25 * far[..., None])
    mask = (ys >= crest[None, :]).astype(np.float32)
    # anti-alias crest edge
    mask = np.clip(ys - crest[None, :] + 1, 0, 1)
    region = frame[y0:]
    region[:] = region * (1 - mask[..., None]) + col * mask[..., None]
    # foreground tall blades (fast, blurred) for parallax punch
    return frame


def tall_grass(img, t, speed):
    d = ImageDraw.Draw(img, 'RGBA')
    r = np.random.default_rng(int(t * 1000) % 7 + 50)
    rr = np.random.default_rng(77)
    for k in range(34):
        base_x = rr.uniform(-100, W + 100)
        phase = rr.uniform(0, 1)
        side = 1 if base_x > W * 0.52 else -1
        # blades rush outward and down as we move forward
        p = (phase + t * speed * rr.uniform(0.6, 1.2)) % 1.0
        x = W * 0.52 + (base_x - W * 0.52) * (0.6 + 1.6 * p)
        y = 800 + 320 * p ** 1.5
        h = 40 + 220 * p ** 2
        lean = side * (10 + 50 * p)
        c = mix((90, 150, 40), (210, 240, 110), rr.random())
        a = int(255 * min(1, p * 3))
        wdt = max(3, int(4 + 10 * p))
        d.polygon([(x - wdt, y), (x + wdt, y), (x + lean, y - h)], fill=c + (int(a * 0.85),))


# ---------------------------------------------------------------- character
SP = 3  # sprite supersample


def draw_rider(t, pedal_hz=1.6):
    """Original cartoon bear cub on a teal bike, seen from behind 3/4. Returns RGBA sprite."""
    sw, sh = 420, 560
    img = Image.new('RGBA', (sw * SP, sh * SP), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    S = lambda *p: [v * SP for v in p]
    OUT = (28, 22, 30, 255)
    ow = 5 * SP
    ang = 2 * math.pi * pedal_hz * t

    def wheel(cx, cy, rx, ry, spin):
        d.ellipse(S(cx - rx, cy - ry, cx + rx, cy + ry), outline=OUT, width=int(13 * SP))
        d.ellipse(S(cx - rx + 9, cy - ry + 9, cx + rx - 9, cy + ry - 9), outline=(70, 70, 80, 255), width=int(3 * SP))
        for k in range(8):
            a = spin + k * math.pi / 8
            d.line(S(cx, cy, cx + (rx - 10) * math.cos(a), cy + (ry - 10) * math.sin(a)), fill=(200, 205, 215, 200), width=SP)
            d.line(S(cx, cy, cx - (rx - 10) * math.cos(a), cy - (ry - 10) * math.sin(a)), fill=(200, 205, 215, 200), width=SP)
        d.ellipse(S(cx - 9, cy - 9, cx + 9, cy + 9), fill=(150, 155, 165, 255), outline=OUT, width=2 * SP)

    spin = ang * 2.6
    # front wheel (further, up-right) then rear wheel (near, big)
    wheel(300, 405, 62, 112, spin)
    TEAL = (24, 178, 170, 255)
    # frame
    d.line(S(196, 470, 300, 330), fill=OUT, width=int(17 * SP))
    d.line(S(196, 470, 300, 330), fill=TEAL, width=int(11 * SP))
    d.line(S(196, 470, 196, 300), fill=OUT, width=int(17 * SP))
    d.line(S(196, 470, 196, 300), fill=TEAL, width=int(11 * SP))
    d.line(S(300, 330, 300, 405), fill=OUT, width=int(13 * SP))
    # handlebars
    d.line(S(236, 292, 352, 282), fill=OUT, width=int(10 * SP))
    wheel(196, 452, 70, 104, spin + 0.4)
    # legs: pedalling (left leg, right leg out of phase)
    for ph, side, shade in [(0, -1, (150, 96, 58)), (math.pi, 1, (176, 116, 70))]:
        a = ang + ph
        hip = (196 + side * 40, 300)
        foot = (196 + side * 40 + 26 * math.cos(a), 430 + 34 * math.sin(a))
        knee = (hip[0] + side * 18 + 0.35 * (foot[0] - hip[0]), (hip[1] + foot[1]) / 2 - 18 + 10 * math.sin(a))
        for seg in [(hip, knee), (knee, foot)]:
            d.line(S(*seg[0], *seg[1]), fill=OUT, width=int(40 * SP))
            d.line(S(*seg[0], *seg[1]), fill=shade + (255,), width=int(31 * SP))
        d.ellipse(S(knee[0] - 15, knee[1] - 15, knee[0] + 15, knee[1] + 15), fill=shade + (255,))
        # sneaker
        fx, fy = foot
        d.ellipse(S(fx - 22, fy - 14, fx + 22, fy + 14), fill=(240, 60, 60, 255), outline=OUT, width=ow)
        d.ellipse(S(fx - 18, fy + 2, fx + 18, fy + 12), fill=(255, 255, 255, 255))
    # shorts
    d.rounded_rectangle(S(124, 222, 270, 326), radius=30 * SP, fill=(30, 34, 52, 255), outline=OUT, width=ow)
    # body: white T-shirt (sunlit -> HDR highlight), leaning forward
    d.ellipse(S(108, 98, 288, 272), fill=(222, 226, 238, 255), outline=OUT, width=ow)
    d.chord(S(108, 98, 288, 272), 100, 250, fill=(196, 204, 226, 255))   # shade side
    d.ellipse(S(108, 98, 288, 272), outline=OUT, width=ow)
    d.ellipse(S(196, 124, 250, 170), fill=(255, 255, 255, 255))  # sun hit on shoulder
    # arms reaching to bars
    for sx, sy, ex, ey, hx, hy, c in [(124, 150, 100, 222, 238, 292, (150, 96, 58)),
                                      (272, 146, 312, 214, 350, 282, (176, 116, 70))]:
        for seg in [((sx, sy), (ex, ey)), ((ex, ey), (hx, hy))]:
            d.line(S(*seg[0], *seg[1]), fill=OUT, width=int(30 * SP))
            d.line(S(*seg[0], *seg[1]), fill=c + (255,), width=int(22 * SP))
        d.ellipse(S(ex - 13, ey - 13, ex + 13, ey + 13), fill=c + (255,))
        d.ellipse(S(hx - 14, hy - 14, hx + 14, hy + 14), fill=c + (255,), outline=OUT, width=ow)
    # short sleeves over the shoulders
    for cx_, cy_ in [(124, 160), (272, 154)]:
        d.ellipse(S(cx_ - 26, cy_ - 24, cx_ + 26, cy_ + 24), fill=(232, 236, 246, 255), outline=OUT, width=ow)
    # head: bear cub, back of head, round ears, red helmet
    hx, hy = 206, 70 + 4 * math.sin(ang * 2)
    for ex in (-58, 58):
        d.ellipse(S(hx + ex - 26, hy - 50, hx + ex + 26, hy + 2), fill=(170, 108, 62, 255), outline=OUT, width=ow)
        d.ellipse(S(hx + ex - 13, hy - 37, hx + ex + 13, hy - 11), fill=(232, 170, 120, 255))
    d.ellipse(S(hx - 72, hy - 58, hx + 72, hy + 70), fill=(176, 114, 66, 255), outline=OUT, width=ow)
    d.ellipse(S(hx - 20, hy - 50, hx + 56, hy + 6), fill=(214, 150, 92, 255))  # sun on fur
    d.chord(S(hx - 74, hy - 66, hx + 74, hy + 40), 180, 360, fill=(226, 40, 46, 255), outline=OUT, width=ow)
    d.line(S(hx - 74, hy - 13, hx + 74, hy - 13), fill=OUT, width=ow)
    d.arc(S(hx - 40, hy - 58, hx + 50, hy - 6), 210, 320, fill=(255, 190, 190, 255), width=int(8 * SP))  # gloss
    for vx in (-30, 0, 30):
        d.line(S(hx + vx, hy - 62, hx + vx * 1.1, hy - 20), fill=(150, 18, 24, 255), width=int(5 * SP))
    # little tail
    d.ellipse(S(182, 238, 214, 268), fill=(176, 114, 66, 255), outline=OUT, width=ow)
    return img.resize((sw, sh), Image.LANCZOS)


# ---------------------------------------------------------------- post
def bloom(arr):
    """Soft glow from the brightest areas (sells the HDR highlights)."""
    lum = arr.max(axis=2)
    bright = np.clip((lum - 215) / 40, 0, 1)[..., None] * arr
    small = Image.fromarray(bright.astype(np.uint8)).resize((W // 8, H // 8), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(6))
    glow = np.asarray(small.resize((W, H), Image.BILINEAR)).astype(np.float32)
    return arr + 0.35 * glow


def vignette():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W * 0.7)) ** 2 + ((y - H / 2) / (H * 0.75)) ** 2)
    return (1 - 0.22 * np.clip(r - 0.35, 0, 1) ** 1.5)[..., None]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--seconds', type=float, default=7.0)
    ap.add_argument('--preview', type=float, nargs='*', help='write PNG stills at these times instead')
    a = ap.parse_args()

    LW = 3400
    sky = make_sky(W + 200)
    clouds = make_clouds(LW)
    far = make_far(LW)
    city, deck = make_city(LW)
    trees = make_trees(LW)
    vig = vignette()

    def frame_at(t):
        T = a.seconds
        k = t / T
        pan = 380 * k + 40 * math.sin(k * math.pi)       # camera yaw drift to the right
        bob = 6 * math.sin(2 * math.pi * 3.2 * t) + 3 * math.sin(2 * math.pi * 1.3 * t)
        roll_x = 10 * math.sin(2 * math.pi * 0.4 * t)
        fr = sky.crop((int(pan * 0.02), 0, int(pan * 0.02) + W, H)).copy()
        def put(layer, px, py):
            fr.alpha_composite(layer, (int(-px), int(py)))
        put(clouds, 200 + pan * 0.08 + t * 12, 0 + bob * 0.1)
        put(far, 300 + pan * 0.18 + roll_x * 0.2, -10 + bob * 0.2)
        put(city, 340 + pan * 0.45 + roll_x * 0.45, 30 + bob * 0.45)
        # train on the viaduct deck, moving right
        dtr = ImageDraw.Draw(fr)
        ox = 340 + pan * 0.45 + roll_x * 0.45
        oy = 30 + bob * 0.45
        tx0 = 1150 + 110 * t
        for c in range(6):
            cx = tx0 + c * 74
            y1, y2 = deck(cx), deck(cx + 66)
            sx1, sx2 = cx - ox, cx + 66 - ox
            pts = [(sx1, y1 - 26 + oy), (sx2, y2 - 26 + oy), (sx2, y2 + 2 + oy), (sx1, y1 + 2 + oy)]
            dtr.polygon(pts, fill=(36, 70, 170), outline=(16, 20, 40))
            dtr.line([(sx1 + 4, y1 - 16 + oy), (sx2 - 4, y2 - 16 + oy)], fill=(255, 236, 170), width=5)
        put(trees, 380 + pan * 0.7 + roll_x * 0.7, 60 + bob * 0.7)
        arr = np.asarray(fr.convert('RGB')).astype(np.float32)
        # crest of the grassy slope (wavy line, slowly drifting)
        xs = np.arange(W)
        crest = (760 + 40 * np.sin(xs * 0.0021 + 0.6 + k * 1.3) + 26 * np.sin(xs * 0.0063 + 2 + k * 2)
                 - 60 * (xs / W) + bob * 1.0).astype(np.float32)
        # rider: drifts from left-centre to centre, grows a bit (camera catching up)
        rider = draw_rider(t)
        sc = 0.95 + 0.25 * k
        rw, rh = int(rider.width * sc), int(rider.height * sc)
        rider = rider.resize((rw, rh), Image.LANCZOS)
        rx = int(W * (0.36 + 0.12 * k) + 30 * math.sin(t * 1.1) - rw / 2)
        render_ground(arr, t, cam_z=18 * t, bob=bob, crest=crest,
                      path_x0=rx + rw * 0.47, path_x1=W * 0.62 - pan * 0.3)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert('RGBA')
        ry = int(H - rh - 70 + bob * 0.3 + 6 * math.sin(2 * math.pi * 1.6 * t))
        # contact shadow
        sh = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(sh).ellipse([rx + rw * 0.25, ry + rh * 0.86, rx + rw * 0.85, ry + rh * 1.02], fill=(10, 40, 0, 120))
        img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)))
        img.alpha_composite(rider, (rx, ry))
        tall_grass(img, t, speed=0.9)
        arr = np.asarray(img.convert('RGB')).astype(np.float32)
        arr = bloom(arr) * vig
        return np.clip(arr, 0, 255).astype(np.uint8)

    if a.preview is not None:
        for t in a.preview:
            Image.fromarray(frame_at(t)).save(f'{a.out}_{t:.1f}.png')
        return

    n = int(round(a.seconds * a.fps))
    p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                          '-r', str(a.fps), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '12',
                          '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                          '-colorspace', 'bt709', a.out], stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(frame_at(i / a.fps).tobytes())
        if i % 60 == 0:
            print(f'frame {i}/{n}', file=sys.stderr, flush=True)
    p.stdin.close(); p.wait()


if __name__ == '__main__':
    main()
