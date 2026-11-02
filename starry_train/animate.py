"""Animate one still (night sky, clouds, steam train) into a seamless HDR (HLG) loop.

Everything that moves is periodic in the loop length, so the clip loops without a seam:
twinkling stars, a pulsing four-point star, shooting stars, lightning inside the right cloud bank,
a flickering headlight and station windows, and grass swaying in the foreground.

HDR grade: the scene stays at SDR level; point lights (stars, lamps, lightning) are lifted to
~1000 nits and the lit cloud tops get a gentle lift, so the highlights are small and sit on a dark sky.

    python3 animate.py still.jpg loop.mp4 [--seconds 20] [--fps 30] [--preview T ...]
"""
import argparse
import math
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from grade_pop import hlg_from_nits  # noqa: E402
from pfp_hdr import M709_TO_2020, SDR_WHITE, smoothstep, srgb_to_linear  # noqa: E402


def value_noise(h, w, cell, seed):
    """Blocky random field in [0,1): one value per cell, so each small star gets one value."""
    g = np.random.default_rng(seed).random((h // cell + 1, w // cell + 1)).astype(np.float32)
    return np.repeat(np.repeat(g, cell, 0), cell, 1)[:h, :w]

W, H = 1080, 1920
LUMA = np.array([0.2627, 0.6780, 0.0593], np.float32)


def load(path):
    im = Image.open(path).convert('RGB')
    s = W / im.width
    im = im.resize((W, round(im.height * s)), Image.LANCZOS)
    top = (im.height - H) // 2
    return np.asarray(im.crop((0, top, W, top + H))).astype(np.float32) / 255, s, top


def _box(a, r, axis):
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim; pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode='edge'), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def blur(a, r):
    """Approximate Gaussian blur (three box passes per axis) for float arrays."""
    b = max(1, int(round(r * 0.8))) if r >= 1 else 0
    a = a.astype(np.float32)
    for _ in range(3):
        a = _box(_box(a, b, 0), b, 1)
    return a


def glow_sprite(radius, falloff):
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    return np.exp(-np.hypot(x, y) / falloff)


def add(dst, sprite, cx, cy, colour, amount):
    r = sprite.shape[0] // 2
    x0, y0 = int(cx) - r, int(cy) - r
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + sprite.shape[1], W), min(y0 + sprite.shape[0], H)
    if xa >= xb or ya >= yb:
        return
    s = sprite[ya - y0:yb - y0, xa - x0:xb - x0]
    dst[ya:yb, xa:xb] += s[..., None] * (np.array(colour, np.float32) * amount)


# Train outline, wheel base and horizon, in frame coordinates (1080x1920).
TRAIN_POLY = [(358, 1384), (474, 1329), (526, 1303), (561, 1301), (567, 1324), (606, 1338), (615, 1345),
              (615, 1456), (592, 1471), (588, 1499), (510, 1499), (503, 1479), (480, 1476), (358, 1432)]
ANCHOR = (552.0, 1498.0)      # bottom centre of the locomotive, on the tracks
HORIZON = 1413.0
LAMP_REL = (-2.0, -168.0)     # headlight relative to ANCHOR at scale 1


def cut_train(srgb):
    """Remove the train from the still. Returns (filled background, RGBA sprite in sRGB with its origin)."""
    from PIL import ImageDraw
    m = Image.new('L', (W, H), 0)
    ImageDraw.Draw(m).polygon(TRAIN_POLY, fill=255)
    hard = np.asarray(m).astype(np.float32) / 255
    soft = np.clip(blur(hard, 1.5) * 1.15, 0, 1)
    # fill: the same rows 260 px to the right (open sky, horizon and grass), mirrored for a clean seam
    fill = srgb.copy()
    ys, xs = np.nonzero(blur(hard, 3) > 0.01)
    x0, x1 = xs.min(), xs.max()
    for x in range(x0, x1 + 1):
        fill[:, x] = srgb[:, x1 + 1 + (x1 - x)]
    # rails continue to the horizon where the locomotive stood
    from PIL import ImageDraw as D
    rails = Image.new('L', (W, H), 0); d = D.Draw(rails)
    for xb in (521, 589):
        d.line([(xb, ANCHOR[1] + 4), (553 + (xb - 553) * 0.12, HORIZON + 2)], fill=255, width=3)
    r = blur(np.asarray(rails).astype(np.float32) / 255, 0.7)[..., None] * blur(hard, 3)[..., None]
    fill = fill * (1 - r) + np.array([0.34, 0.27, 0.22], np.float32) * r
    out = srgb * (1 - soft[..., None]) + fill * soft[..., None]
    ys, xs = np.nonzero(hard > 0)
    bx0, bx1, by0, by1 = xs.min() - 3, xs.max() + 4, ys.min() - 3, ys.max() + 4
    spr = np.dstack([srgb[by0:by1, bx0:bx1], np.clip(blur(hard, 0.8), 0, 1)[by0:by1, bx0:bx1]])
    return out, (spr, bx0, by0)


def train_state(t, T):
    """Scale (1 = parked at the station) and fade of the train at time t; None when no train is visible."""
    t %= T
    d0, d1 = 0.30 * T, 0.43 * T            # departure toward the camera
    a0 = 0.70 * T                          # a new train arrives from the horizon and stops at T
    if t < d0:
        return 1.0, 1.0, 0.0
    if t < d1:
        u = (t - d0) / (d1 - d0)
        z = 1 - 0.93 * u ** 1.7
        return 1 / z, 1.0 - smoothstep(0.975, 1.0, u), u
    if t < a0:
        return None
    u = (t - a0) / (T - a0)
    z = 1 + 9 * (1 - u) ** 2
    return 1 / z, smoothstep(0.0, 0.08, u), 0.0


def place_train(nits, sprite_lin, alpha, sc, fade):
    """Composite the sprite (linear nits, alpha) scaled by sc with its wheels on the track."""
    spr, bx0, by0 = sprite_lin
    h, w = spr.shape[:2]
    nw, nh = max(2, int(round(w * sc))), max(2, int(round(h * sc)))
    if nw > 6 * W:
        return
    ax, ay = ANCHOR
    by = HORIZON + (ay - HORIZON) * sc          # wheels follow the ground plane
    ox = ax + (bx0 - ax) * sc
    oy = by + (by0 - ay) * sc
    rs = lambda ch: np.asarray(Image.fromarray(ch, 'F').resize((nw, nh), Image.BILINEAR))
    rgb = np.dstack([rs(spr[..., c]) for c in range(3)])
    al = rs(alpha) * fade
    x0, y0 = int(round(ox)), int(round(oy))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + nw, W), min(y0 + nh, H)
    if xa >= xb or ya >= yb:
        return
    sub = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    A = al[sub][..., None]
    nits[ya:yb, xa:xb] = nits[ya:yb, xa:xb] * (1 - A) + rgb[sub] * A


_RENDER = None


def _worker(i):
    return _RENDER(i)


def _encode(sig):
    g = np.round(sig * 65535).astype('<u2')
    return np.ascontiguousarray(g[..., [1, 2, 0]].transpose(2, 0, 1)).tobytes()   # planar G, B, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--seconds', type=float, default=20)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--preview', type=float, nargs='*')
    ap.add_argument('--train', action='store_true', help='train departs toward the camera and a new one arrives')
    a = ap.parse_args()
    T = a.seconds

    srgb, s, top = load(a.src)
    P = lambda x, y: (x * s, y * s - top)          # source-image coordinates -> frame
    if a.train:
        srgb, sprite = cut_train(srgb)
    lin = srgb_to_linear(srgb) @ M709_TO_2020.T.astype(np.float32)
    Y = lin @ LUMA
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # --- masks from the still
    local = Y - blur(Y, 6)
    star = smoothstep(0.03, 0.15, local) * (yy < P(0, 1500)[1])          # small bright points in the sky
    phase = value_noise(H, W, 3, 11)                                       # per-star twinkle phase
    speed = np.floor(1 + 3 * value_noise(H, W, 3, 12))                     # 1..3 cycles per loop
    code709 = srgb
    bluish = (code709[..., 2] - code709[..., 0] > 0.12) & (Y > 0.35)
    cloud_r = (xx > W * 0.55) & (yy > P(0, 850)[1]) & (yy < P(0, 1420)[1])
    bolt = blur((bluish & cloud_r).astype(np.float32), 1.0)
    bolt_glow = blur(bolt, 40) * 6
    clouds = smoothstep(0.25, 0.7, Y) * (1 - star) * (yy < P(0, 1450)[1])
    warm_win = ((code709[..., 0] > 0.6) & (code709[..., 0] - code709[..., 2] > 0.25)
                & (xx > P(180, 0)[0]) & (xx < P(420, 0)[0]) & (yy > P(0, 1380)[1]) & (yy < P(0, 1500)[1])).astype(np.float32)

    # base grade: SDR level, gentle lift on lit clouds, strong lift on points of light
    base_gain = 1.0 + 0.7 * clouds + 4.0 * star * smoothstep(0.15, 0.6, Y)

    # grass sway: displacement grows toward the bottom; the tracks stay still
    g0 = P(0, 1520)[1]
    amp = smoothstep(g0, H, yy) * 5.0
    tx_c = W * 0.51
    half = 45 + (yy - g0) / (H - g0) * 210
    rails = 1 - smoothstep(half, half + 40, np.abs(xx - tx_c))
    amp *= (1 - rails) * (yy > g0)

    star_xy = P(565, 605)
    lamp_xy = P(570, 1383)
    if a.train:
        spr, sbx, sby = sprite
        spr_lin = srgb_to_linear(spr[..., :3]) @ M709_TO_2020.T.astype(np.float32) * SDR_WHITE
        train_sprite = ((spr_lin.astype(np.float32), sbx, sby), spr[..., 3].astype(np.float32))
    cross = np.zeros((161, 161), np.float32)
    c = 80
    cross[c, :] = np.exp(-np.abs(np.arange(161) - c) / 26)
    cross[:, c] = np.exp(-np.abs(np.arange(161) - c) / 34)
    cross = blur(cross, 1.2) * 3
    halo = glow_sprite(80, 10)
    lamp = glow_sprite(120, 18)
    # shooting stars: (start time, start x, start y, dx, dy) in frame coords
    rng = np.random.default_rng(4)
    meteors = [(T * f, rng.uniform(150, 950), rng.uniform(60, 500), rng.uniform(-420, -220), rng.uniform(160, 300))
               for f in (0.12, 0.47, 0.81)]
    flashes = [T * f for f in (0.22, 0.25, 0.63, 0.66, 0.68, 0.9)]
    lin_rgb = lin.copy()

    def frame(t):
        k = 2 * math.pi * t / T
        # grass sway (bilinear-free nearest remap is enough at this amplitude)
        st = train_state(t, T) if a.train else (1.0, 1.0, 0.0)
        rush = 0.0 if st is None else smoothstep(0.35, 1.0, st[2])      # 0..1 while the train rushes past
        dx = (amp * (1 + 3 * rush) * np.sin(2 * k + xx * 0.012 + yy * 0.018)
              + amp * 0.4 * np.sin(5 * k + yy * 0.05))
        sx = np.clip(np.rint(xx + dx), 0, W - 1).astype(np.int32)
        src = lin_rgb[yy.astype(np.int32), sx]
        gain = base_gain[yy.astype(np.int32), sx]
        tw = 1 + 0.7 * np.sin(speed * k + 2 * math.pi * phase * 7)
        gain = gain * (1 + star * (tw - 1))
        # lightning: quick flashes with decay, schedule repeats every loop
        f = sum(math.exp(-((t - t0) % T) * 9) for t0 in flashes)
        nits = src * SDR_WHITE * gain[..., None]
        nits += (bolt * 900 * f)[..., None] * np.array([0.75, 0.85, 1.0], np.float32)
        nits += (bolt_glow * 60 * f)[..., None] * np.array([0.7, 0.8, 1.0], np.float32)
        # windows and headlight flicker
        fl = 1 + 0.08 * math.sin(3 * k) + 0.05 * math.sin(11 * k)
        nits += (warm_win * 250 * (fl - 0.9))[..., None] * np.array([1.0, 0.75, 0.4], np.float32)
        if not a.train:
            add(nits, lamp, *lamp_xy, (1.0, 0.85, 0.6), 520 * fl)
            add(nits, halo, *lamp_xy, (1.0, 0.95, 0.85), 900 * fl)
        elif st is not None:
            sc, fade, _ = st
            # motion blur: average a few positions across half a frame while moving fast
            nxt = train_state(t + 0.5 / a.fps, T)
            sc2 = nxt[0] if nxt is not None else sc
            samples = 1 if abs(sc2 - sc) < 0.01 * sc else 6
            acc = np.zeros_like(nits)
            for j in range(samples):
                sj = sc + (sc2 - sc) * j / max(samples - 1, 1)
                layer = nits.copy()
                place_train(layer, *train_sprite, sj, fade)
                acc += layer
            nits = acc / samples
            lx = ANCHOR[0] + LAMP_REL[0] * sc
            ly = HORIZON + (ANCHOR[1] - HORIZON) * sc + LAMP_REL[1] * sc
            gr = min(sc, 9.0)
            big = glow_sprite(int(min(120 * gr, 900)), 18 * gr)
            add(nits, big, lx, ly, (1.0, 0.85, 0.6), 520 * fl * fade * gr ** 0.6)
            add(nits, glow_sprite(int(min(80 * gr, 700)), 10 * gr), lx, ly, (1.0, 0.95, 0.85), 900 * fl * fade)
        if a.train and st is not None and st[2] > 0:
            u = st[2]
            wash = math.exp(-((1 - u) * 12) ** 2)                         # warm light as it passes the camera
            nits += np.array([60, 48, 32], np.float32) * wash
            sh = int(round(10 * rush * math.sin(t * 61)))
            sv = int(round(7 * rush * math.sin(t * 47 + 1)))
            nits = np.pad(nits, ((12, 12), (12, 12), (0, 0)), mode='edge')[12 - sv:12 - sv + H, 12 - sh:12 - sh + W]
        # the four-point star: pulsing core and spikes
        pulse = 1 + 0.35 * math.sin(k * 3)
        add(nits, cross, *star_xy, (0.75, 0.88, 1.0), 380 * pulse)
        add(nits, halo, *star_xy, (0.8, 0.9, 1.0), 1100 * pulse)
        # shooting stars
        for t0, x0, y0, mdx, mdy in meteors:
            u = ((t - t0) % T) / 0.9
            if u < 1:
                for j in range(14):
                    v = u - j * 0.018
                    if 0 <= v <= 1:
                        add(nits, glow_sprite(6, 1.6), x0 + mdx * v, y0 + mdy * v, (0.85, 0.92, 1.0),
                            1200 * (1 - j / 14) * math.sin(math.pi * u))
        return nits

    if a.preview is not None:
        for t in a.preview:
            n = frame(t)
            sdr = np.clip((n @ np.linalg.inv(M709_TO_2020).T) / SDR_WHITE, 0, 1)
            sdr = np.where(sdr <= 0.0031308, sdr * 12.92, 1.055 * sdr ** (1 / 2.4) - 0.055)
            Image.fromarray((sdr * 255).astype(np.uint8)).save(f'{a.out}_{t:.1f}.png')
            l = n @ LUMA
            print(f't={t:.1f}  median {np.median(l):.0f} nits  p99 {np.percentile(l, 99):.0f}  max {l.max():.0f}  '
                  f'>400 {np.mean(l > 400) * 100:.2f}%')
        return

    enc = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gbrp16le', '-s', f'{W}x{H}', '-r', str(a.fps),
        '-i', '-', '-vf', 'zscale=min=gbr:m=bt2020nc:r=tv:rin=full,format=yuv420p',
        '-c:v', 'libx264', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-preset', 'slow',
        '-b:v', '10M', '-maxrate', '12M', '-bufsize', '20M', '-g', str(a.fps * 2),
        '-color_primaries', 'bt2020', '-color_trc', 'arib-std-b67', '-colorspace', 'bt2020nc', '-color_range', 'tv',
        '-x264-params', 'colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc',
        '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
    n = int(round(T * a.fps))
    global _RENDER
    _RENDER = lambda i: _encode(hlg_from_nits(frame(i / a.fps)))
    with mp.get_context('fork').Pool(os.cpu_count()) as pool:
        for i, buf in enumerate(pool.imap(_worker, range(n), chunksize=2)):
            enc.stdin.write(buf)
            if i % 60 == 0:
                print(f'frame {i}/{n}', file=sys.stderr, flush=True)
    enc.stdin.close(); enc.wait()


if __name__ == '__main__':
    main()
