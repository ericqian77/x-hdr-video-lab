#!/usr/bin/env python3
"""Turn a still image into a short, seamlessly looping SDR clip whose motion and
light effects are built to "pop" after grade_hlg.sh turns it into HLG HDR.

Effects (all composed in linear light):
  * slow orbit camera: breathing push-in, small roll, floating drift
  * pulsing bloom + white-hot cores on the brightest parts (lamps, windows, glass)
  * a glossy light sweep across bright, low-saturation surfaces (titles, plates, white)
  * small cool-white sparkles / dust motes with a few star glints

Small, pure-white highlights over a darker frame are what read as "HDR" on an
iPhone, so the effects add exactly that instead of lifting the whole image.

Usage:
  python3 still_to_motion.py in.jpg sdr.mp4 [--seconds 8] [--fps 30] [--width 1080]
  SIZE=1080:1620 PRE="eq=brightness=0.02," bash grade_hlg.sh sdr.mp4 hdr.mp4

Needs numpy, Pillow and ffmpeg.
"""
import argparse
import math
import subprocess
import sys

import numpy as np
from PIL import Image


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def box_blur(img, r):
    """Separable box blur with edge clamping; three passes approximate a Gaussian."""
    if r < 1:
        return img
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r + 1, r)
        p = np.pad(img, pad, mode="edge")
        c = np.cumsum(p, axis=axis, dtype=np.float32)
        n = img.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        img = (hi - lo) / (2 * r + 1)
    return img


def gauss_blur(img, sigma):
    r = max(1, int(round(sigma * 0.85)))
    for _ in range(3):
        img = box_blur(img, r)
    return img


def resize_f(img, w, h):
    """Resize an HxWxC float image with Pillow, one channel at a time."""
    chans = [np.asarray(Image.fromarray(img[..., c].astype(np.float32), "F").resize((w, h), Image.BILINEAR))
             for c in range(img.shape[2])]
    return np.stack(chans, axis=2)


class Sparkles:
    def __init__(self, w, h, n, seed=7):
        rng = np.random.default_rng(seed)
        self.w, self.h = w, h
        self.n = n
        self.u = rng.uniform(0.03, 0.97, n)
        self.v = rng.uniform(0.03, 0.97, n)
        # integer multipliers keep every motion periodic over one loop
        self.rise = rng.integers(1, 3, n) * rng.choice([1, 1, 1, -1], n)
        self.wob_k = rng.integers(1, 4, n)
        self.wob_a = rng.uniform(0.004, 0.02, n)
        self.ph = rng.uniform(0, 2 * math.pi, n)
        self.tw_k = rng.integers(2, 6, n)
        self.size = rng.choice([0.9, 1.2, 1.6, 2.2, 3.0], n, p=[0.3, 0.3, 0.2, 0.13, 0.07]) * (w / 1080)
        self.gain = rng.uniform(0.8, 2.5, n)
        warm = rng.random(n) < 0.3
        self.col = np.where(warm[:, None], [1.0, 0.88, 0.65], [0.85, 0.93, 1.0]).astype(np.float32)
        self.star = rng.random(n) < 0.12
        self.bokeh = rng.random(n) < 0.08

    def render(self, out, phase):
        h, w = self.h, self.w
        for i in range(self.n):
            vv = (self.v[i] - self.rise[i] * phase / (2 * math.pi)) % 1.0
            uu = self.u[i] + self.wob_a[i] * math.sin(self.wob_k[i] * phase + self.ph[i])
            x, y = uu * w, vv * h
            tw = 0.5 + 0.5 * math.sin(self.tw_k[i] * phase + self.ph[i] * 3)
            tw = tw ** 3
            s = self.size[i] * (3.5 if self.bokeh[i] else 1.0)
            amp = self.gain[i] * tw * (0.25 if self.bokeh[i] else 1.0)
            if amp < 0.02:
                continue
            rad = int(math.ceil(s * 3 + (s * 14 if self.star[i] else 0)))
            x0, x1 = max(0, int(x) - rad), min(w, int(x) + rad + 1)
            y0, y1 = max(0, int(y) - rad), min(h, int(y) + rad + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dx, dy = xx - x, yy - y
            k = np.exp(-(dx * dx + dy * dy) / (2 * s * s))
            if self.star[i]:
                L = s * 12
                k = k + 0.55 * tw * (np.exp(-(dy * dy) / (0.5 * s * s)) * np.exp(-np.abs(dx) / L * 3)
                                     + np.exp(-(dx * dx) / (0.5 * s * s)) * np.exp(-np.abs(dy) / L * 3))
            out[y0:y1, x0:x1] += (amp * k)[..., None] * self.col[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1080, help="output width; height follows the image aspect")
    ap.add_argument("--zoom", type=float, default=1.075, help="mean push-in (needs >= ~1.05 to hide corners)")
    ap.add_argument("--roll", type=float, default=1.0, help="max roll in degrees")
    ap.add_argument("--bloom", type=float, default=1.0, help="bloom strength")
    ap.add_argument("--sweep", type=float, default=1.0, help="light sweep strength (0 = off)")
    ap.add_argument("--sweep-box", type=str, default="0,0,1,1",
                    help="x0,y0,x1,y1 (0-1) limiting the sweep, e.g. to a title; skin looks odd when swept")
    ap.add_argument("--sparkles", type=int, default=90, help="number of sparkles (0 = off)")
    ap.add_argument("--frames", type=str, default="", help="debug: comma list of frame indices to save as PNG")
    a = ap.parse_args()

    src = Image.open(a.src).convert("RGB")
    W = a.width & ~1
    H = int(round(W * src.height / src.width)) & ~1
    # supersample the source so the moving camera stays sharp
    big = src.resize((W * 2, H * 2), Image.LANCZOS)
    bw, bh = big.size
    N = int(round(a.seconds * a.fps))
    sparkles = Sparkles(W, H, a.sparkles) if a.sparkles else None
    debug = {int(x) for x in a.frames.split(",") if x}

    enc = None
    if not debug:
        enc = subprocess.Popen([
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
            "-r", str(a.fps), "-i", "-",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "slow", "-crf", "10",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
            "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn, yn = xx / W, yy / H
    sw, sh = W // 4, H // 4
    bx0, by0, bx1, by1 = (float(v) for v in a.sweep_box.split(","))
    fe = 0.04  # soft edge
    box = (smoothstep(bx0 - fe, bx0 + fe, xn) * (1 - smoothstep(bx1 - fe, bx1 + fe, xn))
           * smoothstep(by0 - fe, by0 + fe, yn) * (1 - smoothstep(by1 - fe, by1 + fe, yn)))
    if bx0 <= 0 and by0 <= 0 and bx1 >= 1 and by1 >= 1:
        box = np.ones_like(xn)

    for f in range(N):
        if debug and f not in debug:
            continue
        t = f / N
        ph = 2 * math.pi * t

        # ---- orbit camera (all terms periodic -> seamless loop) ----
        z = a.zoom + 0.025 * math.sin(ph)
        rot = math.radians(a.roll) * math.sin(ph + 0.6)
        dx = 0.006 * math.sin(ph + math.pi / 2)
        dy = 0.006 * math.sin(2 * ph)
        s = bw / W / z  # output px -> big px
        c, sn = math.cos(rot), math.sin(rot)
        cx, cy = bw / 2 + dx * bw, bh / 2 + dy * bh
        # inverse affine: big = M @ (out - centre) + c
        A, B = s * c, -s * sn
        D, E = s * sn, s * c
        C = cx - A * W / 2 - B * H / 2
        F = cy - D * W / 2 - E * H / 2
        frame = big.transform((W, H), Image.AFFINE, (A, B, C, D, E, F), resample=Image.BICUBIC)
        srgb = np.asarray(frame, dtype=np.float32) / 255.0
        lin = srgb_to_lin(srgb).astype(np.float32)
        Ls = srgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        mx, mn = srgb.max(2), srgb.min(2)
        sat = (mx - mn) / (mx + 1e-4)

        pulse = 0.5 + 0.5 * math.sin(2 * ph - 0.4)          # two breaths per loop
        flick = 1 + 0.06 * math.sin(7 * ph) * math.sin(3 * ph + 1)

        # ---- bloom + white-hot cores on the brightest areas ----
        # skin and warm mids are saturated; light sources and white surfaces are not
        neutral = 1 - smoothstep(0.22, 0.5, sat)
        hl = smoothstep(0.78, 0.97, Ls) * (0.35 + 0.65 * neutral)
        small = resize_f((lin * hl[..., None]).astype(np.float32), sw, sh)
        glow = 0.45 * gauss_blur(small, 2) + 0.3 * gauss_blur(small, 8) + 0.15 * gauss_blur(small, 22)
        glow = resize_f(glow, W, H)
        core = (hl * neutral) ** 2
        lin = lin + core[..., None] * (1.0 - lin) * (0.2 + 0.3 * pulse)
        lin = lin * (1 + core[..., None] * (0.3 + 0.5 * pulse) * flick)
        lin = lin + glow * a.bloom * (0.25 + 0.35 * pulse) * flick

        # ---- glossy sweep across bright, low-saturation surfaces ----
        if a.sweep > 0:
            st = (t * 1.0) % 1.0
            if st < 0.45:
                pos = -0.25 + 1.75 * (st / 0.45)
                d = (xn * 0.8 + yn * 0.6) - pos
                band = np.exp(-(d * d) / (2 * 0.035 ** 2)) + 0.35 * np.exp(-(d * d) / (2 * 0.11 ** 2))
                # bright detail that stands out from its surroundings (lettering, rims, glass),
                # not large smooth bright areas such as skin
                local = resize_f(gauss_blur(resize_f(Ls[..., None], sw, sh), 6), W, H)[..., 0]
                surf = smoothstep(0.6, 0.85, Ls) * smoothstep(0.05, 0.16, Ls - local)
                # skin keeps green well below red (G/R < ~0.87); cream text and white surfaces don't
                gr = srgb[..., 1] / (srgb[..., 0] + 1e-4)
                surf = surf * smoothstep(0.855, 0.885, gr) * box
                # keep solid strokes (lettering, rims) and drop sparse speckle such as skin texture
                dens = gauss_blur(surf, 3)
                surf = surf * smoothstep(0.3, 0.55, dens)
                lin = lin + (a.sweep * 1.6 * band * surf)[..., None] * np.array([1.0, 0.97, 0.92], np.float32)

        # ---- sparkles ----
        if sparkles is not None:
            add = np.zeros_like(lin)
            sparkles.render(add, ph)
            lin = lin + add

        out = (lin_to_srgb(lin) * 255 + 0.5).astype(np.uint8)
        if debug:
            Image.fromarray(out).save(f"{a.out}.{f:04d}.png")
        else:
            enc.stdin.write(out.tobytes())
        if f % 30 == 0:
            print(f"frame {f}/{N}", file=sys.stderr)

    if enc:
        enc.stdin.close()
        sys.exit(enc.wait())


if __name__ == "__main__":
    main()
