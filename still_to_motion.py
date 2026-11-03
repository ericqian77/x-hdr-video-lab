#!/usr/bin/env python3
"""Turn a still image into a short, seamlessly looping SDR clip whose motion and
light effects are built to "pop" after grade_hlg.sh turns it into HLG HDR.

Effects (all composed in linear light):
  * slow orbit camera: breathing push-in, small roll, floating drift
  * pulsing bloom + white-hot cores on the brightest parts (lamps, windows, glass)
  * glow on coloured light sources (lanterns, glowing charms)
  * a glossy light sweep across bright lettering and rims (skin is left alone)
  * small cool-white sparkles / dust motes with a few star glints
  * optional: sparse snow drifting down and up at several depths, animated light rays
  * optional, with --mask: the subject is cut out and floats on its own layer in
    front of the background, with a pulsing rim light, so it lifts off the frame

Small, pure-white highlights over a darker frame are what read as "HDR" on an
iPhone, so the effects add exactly that instead of lifting the whole image.

Usage:
  python3 still_to_motion.py in.jpg sdr.mp4 [--seconds 8] [--fps 30] [--width 1080]
  python3 still_to_motion.py in.png sdr.mp4 --mask subject_mask.png --snow 60 --sparkles 30 --rays 0.6,0
  SIZE=1080:1620 PRE="eq=brightness=0.02," bash grade_hlg.sh sdr.mp4 hdr.mp4
  # stronger HDR contrast: keep the picture at reference white, only lights hit peak
  python3 still_to_motion.py in.png sdr.mp4 --mask m.png --headroom 0.3 --exposure 0.8 ...
  SIZE=1080:1620 CURVE="0/0 1/1" bash grade_hlg.sh sdr.mp4 hdr.mp4

The mask is a greyscale image the size of the source, white = subject (for example
`rembg i -om in.png mask.png`). Needs numpy, Pillow and ffmpeg.
"""
import argparse
import math
import subprocess
import sys

import numpy as np
from PIL import Image

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
COOL = np.array([0.82, 0.92, 1.0], np.float32)


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


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
    """Resize an HxW or HxWxC float image with Pillow, one channel at a time."""
    if img.ndim == 2:
        return np.asarray(Image.fromarray(img.astype(np.float32), "F").resize((w, h), Image.BILINEAR))
    return np.stack([resize_f(img[..., c], w, h) for c in range(img.shape[2])], axis=2)


def push_pull_fill(img, known):
    """Fill pixels where known ~ 0 from their surroundings (cheap inpainting)."""
    levels = [(img * known[..., None], known)]
    while min(levels[-1][1].shape) > 8:
        c, k = levels[-1]
        h, w = k.shape
        levels.append((resize_f(c, max(1, w // 2), max(1, h // 2)), resize_f(k, max(1, w // 2), max(1, h // 2))))
    c, k = levels[-1]
    res = c / np.maximum(k, 1e-6)[..., None]
    for c, k in reversed(levels[:-1]):
        h, w = k.shape
        up = resize_f(res, w, h)
        kk = np.clip(k * 4, 0, 1)[..., None]
        res = c / np.maximum(k, 1e-6)[..., None] * kk + up * (1 - kk)
    return res


def to_img(x):
    x = np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(x)


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
        self.col = np.where(warm[:, None], [1.0, 0.88, 0.65], COOL).astype(np.float32)
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
            splat(out, x, y, s, amp, self.col[i], rad, star=self.star[i], tw=tw)


def splat(out, x, y, s, amp, col, rad, star=False, tw=1.0):
    h, w = out.shape[:2]
    x0, x1 = max(0, int(x) - rad), min(w, int(x) + rad + 1)
    y0, y1 = max(0, int(y) - rad), min(h, int(y) + rad + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - x, yy - y
    k = np.exp(-(dx * dx + dy * dy) / (2 * s * s))
    if star:
        L = s * 12
        k = k + 0.55 * tw * (np.exp(-(dy * dy) / (0.5 * s * s)) * np.exp(-np.abs(dx) / L * 3)
                             + np.exp(-(dx * dx) / (0.5 * s * s)) * np.exp(-np.abs(dy) / L * 3))
    out[y0:y1, x0:x1] += (amp * k)[..., None] * col


class Snow:
    """Sparse snow drifting at several depths. Each flake fades in, drifts a short way
    (most fall, some float up) and fades out again, once per loop, so the clip still
    loops and the frame never fills with noise. Far flakes are small and sharp; near
    flakes are big and soft."""

    def __init__(self, w, h, n, rise=0.3, seed=11):
        rng = np.random.default_rng(seed)
        self.w, self.h, self.n = w, h, n
        self.depth = rng.random(n) ** 1.6          # mostly far flakes
        self.u = rng.uniform(0.02, 0.98, n)
        self.y0 = rng.uniform(0.05, 0.95, n)
        self.v = rng.uniform(0, 1, n)              # where in its life each flake starts
        self.dir = np.where(rng.random(n) < rise, -1.0, 1.0)
        self.travel = rng.uniform(0.18, 0.4, n) * (0.6 + 0.8 * self.depth) * np.where(self.dir < 0, 0.6, 1.0)
        self.slant = rng.uniform(-0.04, 0.08, n)
        self.wob_k = rng.integers(1, 3, n)
        self.wob_a = rng.uniform(0.006, 0.02, n) * (0.6 + self.depth)
        self.ph = rng.uniform(0, 2 * math.pi, n)
        sc = w / 1080
        self.size = (0.8 + 6.0 * self.depth ** 2.2) * sc * rng.uniform(0.8, 1.25, n)
        self.gain = (0.9 + 0.9 * rng.random(n)) * (1.0 - 0.6 * self.depth ** 1.5)

    def render(self, far, near, phase, split=0.55):
        t = phase / (2 * math.pi)
        for i in range(self.n):
            p = (self.v[i] + t) % 1.0
            fade = math.sin(math.pi * p) ** 1.5
            if fade < 0.02:
                continue
            q = p - 0.5
            y = (self.y0[i] + self.dir[i] * self.travel[i] * q) * self.h
            x = (self.u[i] + self.slant[i] * q
                 + self.wob_a[i] * math.sin(self.wob_k[i] * phase + self.ph[i])) * self.w
            s = self.size[i]
            target = near if self.depth[i] > split else far
            splat(target, x, y, s, self.gain[i] * fade, COOL, int(math.ceil(s * 3)))


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
    ap.add_argument("--emissive", type=float, default=1.0, help="glow on coloured light sources (lanterns, charms)")
    ap.add_argument("--sweep", type=float, default=1.0, help="light sweep strength (0 = off)")
    ap.add_argument("--sweep-box", type=str, default="0,0,1,1",
                    help="x0,y0,x1,y1 (0-1) limiting the sweep, e.g. to a title")
    ap.add_argument("--sparkles", type=int, default=90, help="number of sparkles (0 = off)")
    ap.add_argument("--snow", type=int, default=0, help="number of snowflakes (0 = off); ~60 reads as sparse")
    ap.add_argument("--snow-rise", type=float, default=0.3, help="fraction of flakes that float up instead of falling")
    ap.add_argument("--rays", type=str, default="", help="x,y (0-1) of a light source to cast moving rays from")
    ap.add_argument("--rays-strength", type=float, default=1.0)
    ap.add_argument("--mask", type=str, default="", help="greyscale subject matte; makes the subject float")
    ap.add_argument("--float", dest="float_amp", type=float, default=0.012,
                    help="subject float amplitude as a fraction of height (with --mask)")
    ap.add_argument("--rim", type=float, default=1.0, help="rim light on the subject (with --mask)")
    ap.add_argument("--bg-dim", type=float, default=0.85, help="background exposure under the subject layer (with --mask)")
    ap.add_argument("--lift", type=float, default=1.0, help="subject exposure (with --mask)")
    ap.add_argument("--headroom", type=float, default=1.0,
                    help="cap for the base image's whites in linear light (<1 = HDR headroom mode). After a neutral "
                         "HLG grade, SDR ~0.53 (linear ~0.25) lands on reference white, so ~0.3 keeps titles, cloth "
                         "and skin at normal white and leaves the range above to light sources, rim, snow and glints")
    ap.add_argument("--exposure", type=float, default=1.0, help="shadow/midtone gain under the headroom shoulder")
    ap.add_argument("--contrast", type=float, default=1.0, help="power on the base image in linear light (>1 = deeper darks)")
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
    snow = Snow(W, H, a.snow, rise=a.snow_rise) if a.snow else None
    debug = {int(x) for x in a.frames.split(",") if x}

    layered = bool(a.mask)
    if layered:
        alpha = np.asarray(Image.open(a.mask).convert("L").resize((bw, bh), Image.LANCZOS), np.float32) / 255.0
        alpha = smoothstep(0.08, 0.92, alpha)
        # background plate: remove the subject so moving it never reveals a ghost
        qs = (bw // 4, bh // 4)
        grow = np.clip(gauss_blur(resize_f(alpha, *qs), 4) * 4, 0, 1)
        known = 1 - resize_f(grow, bw, bh)
        plate = push_pull_fill(np.asarray(big, np.float32) / 255.0, np.clip(known, 0, 1))
        bg_img = to_img(plate)
        # rim: inner edge of the matte, strongest on edges facing the upper right
        # use a solid silhouette so translucent cloth and loose hair strands don't all get outlined
        k = max(2, bw // 220)
        solid = resize_f(smoothstep(0.35, 0.65, gauss_blur(resize_f(alpha, *qs), 3)), bw, bh)
        top = np.clip(solid - np.roll(solid, (k, -k), axis=(0, 1)), 0, 1)
        allr = np.clip(solid - gauss_blur(solid, k), 0, 1) * 2
        rim = np.clip(gauss_blur(0.75 * top + 0.25 * allr, 1.5) * 1.4, 0, 1) * alpha
        alpha_img, rim_img = to_img(alpha), to_img(rim)
    else:
        bg_img = big

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
    if a.rays:
        rx, ry = (float(v) for v in a.rays.split(","))
        ddx, ddy = (xn - rx) * W / H, yn - ry
        ray_th = np.arctan2(ddy, ddx)
        ray_r = np.sqrt(ddx * ddx + ddy * ddy)
        ray_fall = np.exp(-ray_r / 0.5) * smoothstep(-0.1, 0.5, np.sin(ray_th))
        rng = np.random.default_rng(3)
        ray_f = rng.uniform(18, 70, 6)
        ray_p = rng.uniform(0, 2 * math.pi, 6)
        ray_k = rng.integers(1, 3, 6) * rng.choice([-1, 1], 6)

    def affine(z, rot, dx, dy):
        """Inverse affine (output px -> big px) for PIL, about the frame centre."""
        s = bw / W / z
        c, sn = math.cos(rot), math.sin(rot)
        cx, cy = bw / 2 + dx * bw, bh / 2 + dy * bh
        A, B, D, E = s * c, -s * sn, s * sn, s * c
        return (A, B, cx - A * W / 2 - B * H / 2, D, E, cy - D * W / 2 - E * H / 2)

    def warp(img, m):
        return np.asarray(img.transform((W, H), Image.AFFINE, m, resample=Image.BICUBIC), np.float32) / 255.0

    for f in range(N):
        if debug and f not in debug:
            continue
        t = f / N
        ph = 2 * math.pi * t
        pulse = 0.5 + 0.5 * math.sin(2 * ph - 0.4)          # two breaths per loop
        flick = 1 + 0.06 * math.sin(7 * ph) * math.sin(3 * ph + 1)

        # ---- orbit camera (all terms periodic -> seamless loop) ----
        z = a.zoom + 0.025 * math.sin(ph)
        rot = math.radians(a.roll) * math.sin(ph + 0.6)
        dx = 0.006 * math.sin(ph + math.pi / 2)
        dy = 0.006 * math.sin(2 * ph)
        lin = srgb_to_lin(warp(bg_img, affine(z, rot, dx, dy)))
        alpha_f = None
        if layered:
            # the subject sits nearer the camera: it zooms a little more and floats on its own
            mf = affine(z * 1.025, rot + math.radians(0.6) * math.sin(ph + 1.9),
                        dx + 0.004 * math.sin(ph + 2.4), dy + a.float_amp * math.sin(ph + 0.3))
            fg = srgb_to_lin(warp(big, mf)) * a.lift
            alpha_f = warp(alpha_img, mf)[..., None]   # "L" images warp to HxW
            rim_f = warp(rim_img, mf)
            lin = fg * alpha_f + lin * a.bg_dim * (1 - alpha_f)
            alpha_f = alpha_f[..., 0]

        srgb = lin_to_srgb(lin)
        hr = a.headroom < 1.0
        lin0 = lin
        if hr or a.contrast != 1.0:
            x = np.power(np.clip(lin, 0, 1), a.contrast)
            # soft shoulder: shadows and mids keep their level, whites roll off towards the cap
            lin = a.exposure * x / (1 + a.exposure * x / a.headroom) if hr else x
        Ls = srgb @ LUMA
        mx, mn = srgb.max(2), srgb.min(2)
        sat = (mx - mn) / (mx + 1e-4)
        gr = srgb[..., 1] / (srgb[..., 0] + 1e-4)
        bgw = 1 - alpha_f if layered else 1.0

        # ---- bloom + white-hot cores on the brightest areas ----
        # skin and warm mids are saturated; light sources and white surfaces are not
        neutral = 1 - smoothstep(0.22, 0.5, sat)
        hl = smoothstep(0.86, 0.99, Ls) if hr else smoothstep(0.78, 0.97, Ls)
        hl = hl * (0.35 + 0.65 * neutral)
        if layered:
            hl = hl * (1 - (0.85 if hr else 0.7) * alpha_f)
        # coloured light sources: bright and saturated, but not skin-coloured
        skin = smoothstep(0.76, 0.8, gr) * (1 - smoothstep(0.88, 0.92, gr))
        emis = smoothstep(0.62, 0.92, mx) * smoothstep(0.2, 0.45, sat) * (bgw if layered else 1 - skin)
        src_l = lin0 * (hl + a.emissive * 0.9 * emis)[..., None]
        small = resize_f(src_l.astype(np.float32), sw, sh)
        if hr:
            # tight halo only: a wide glow lifts the darks around every light and reads as fog
            glow = 0.5 * gauss_blur(small, 1.5) + 0.25 * gauss_blur(small, 6)
        else:
            glow = 0.45 * gauss_blur(small, 2) + 0.3 * gauss_blur(small, 8) + 0.15 * gauss_blur(small, 22)
        glow = resize_f(glow, W, H)
        if hr:
            core = 1.6 * hl * neutral + a.emissive * 1.3 * emis
        else:
            core = (hl * neutral) ** 2 + a.emissive * 0.6 * emis ** 2
        core = np.clip(core, 0, 1)
        if hr:
            # light sources go to peak in their own colour (a cyan charm stays cyan)
            target = lin0 / np.maximum(lin0.max(2, keepdims=True), 1e-3)
            lin = lin + core[..., None] * (target - lin) * (0.75 + 0.25 * pulse) * flick
            lin = lin + glow * a.bloom * 0.8 * (0.4 + 0.6 * pulse) * flick
        else:
            lin = lin + core[..., None] * (1.0 - lin) * (0.2 + 0.3 * pulse)
            lin = lin * (1 + core[..., None] * (0.3 + 0.5 * pulse) * flick)
            lin = lin + glow * a.bloom * (0.25 + 0.35 * pulse) * flick

        # ---- moving light rays (behind the subject) ----
        if a.rays:
            beams = np.zeros_like(xn)
            for i in range(6):
                amp = 0.6 + 0.4 * math.sin((i % 3 + 1) * ph + ray_p[i])
                beams += amp * (0.5 + 0.5 * np.cos(ray_f[i] * ray_th + ray_p[i] + ray_k[i] * ph)) ** 6
            rays = a.rays_strength * 0.12 * beams / 3 * ray_fall * (0.8 + 0.2 * pulse)
            if hr:
                rays = rays * a.headroom   # rays are scene light, not a highlight
            lin = lin + (rays * ((1 - 0.85 * alpha_f) if layered else 1.0))[..., None] * COOL

        # ---- glossy sweep across bright lettering and rims ----
        if a.sweep > 0:
            st = (t * 1.0) % 1.0
            if st < 0.45:
                pos = -0.25 + 1.75 * (st / 0.45)
                d = (xn * 0.8 + yn * 0.6) - pos
                band = np.exp(-(d * d) / (2 * 0.035 ** 2)) + 0.35 * np.exp(-(d * d) / (2 * 0.11 ** 2))
                # bright detail that stands out from its surroundings (lettering, rims, glass),
                # not large smooth bright areas such as skin
                local = resize_f(gauss_blur(resize_f(Ls, sw, sh), 6), W, H)
                surf = smoothstep(0.6, 0.85, Ls) * smoothstep(0.05, 0.16, Ls - local)
                # skin keeps green well below red (G/R < ~0.87); cream text and white surfaces don't
                surf = surf * smoothstep(0.855, 0.885, gr) * box
                # keep solid strokes (lettering, rims) and drop sparse speckle such as skin texture
                dens = gauss_blur(surf, 3)
                surf = surf * smoothstep(0.3, 0.55, dens)
                lin = lin + (a.sweep * 1.6 * band * surf)[..., None] * np.array([1.0, 0.97, 0.92], np.float32)

        # ---- rim light: the subject's edge catches the light and lifts off the frame ----
        if layered and a.rim > 0:
            lin = lin + (a.rim * (0.45 if hr else 0.8) * rim_f * (0.55 + 0.45 * pulse))[..., None] * COOL

        # ---- snow and sparkles ----
        if snow is not None:
            far, near = np.zeros_like(lin), np.zeros_like(lin)
            snow.render(far, near, ph)
            gain = 1.5 if hr else 1.0   # flakes are point highlights: let them reach peak
            lin = lin + gain * (far * (bgw[..., None] if layered else 1.0) + near)
        if sparkles is not None:
            add = np.zeros_like(lin)
            sparkles.render(add, ph)
            lin = lin + add * (1.5 if hr else 1.0)

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
