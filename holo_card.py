"""Holographic card: an image as a physical card tilting left and right, with an HDR sheen.

The card rotates in 3D (yaw about +-14 degrees, a little pitch), so it reads as a card held in
the hand. As it turns, a white specular band and a rainbow foil sheen sweep across it, and glitter
points flash, each at its own angle. The artwork stays at SDR level; the sheen and glitter are the
HDR highlights, on a near-black background.
Default style 'glow': no stripe. A wide, soft light lobe multiplies the artwork's own colours as the card
turns toward the light, so saturated colours glow and stay their colour; the bright grain of the texture
shimmers inside the lobe, with sparse glitter. Style 'band' keeps the specular stripe and rainbow foil. Motion is periodic, so the clip loops without a seam.
Output: HLG BT.2020, H.264 High 8-bit, 1080x1920.

    python3 holo_card.py card.jpg out.mp4 [--seconds 6] [--fps 30] [--preview T ...]
"""
import argparse
import math
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from grade_pop import hlg_from_nits
from pfp_hdr import M709_TO_2020, SDR_WHITE, smoothstep, srgb_to_linear

W, H = 1080, 1920
CW, CH = 820, 1640                     # card size in pixels at rest
LUMA = np.array([0.2627, 0.6780, 0.0593], np.float32)
YAW, PITCH = math.radians(14), math.radians(4)


def hsv_rainbow(h):
    """Hue 0..1 -> RGB 0..1, full saturation."""
    k = (np.stack([h, h, h], -1) * 6 + np.array([0, 4, 2])) % 6
    return np.clip(np.minimum(k, 4 - k), 0, 1)


def homography(src, dst):
    """Coefficients for PIL's PERSPECTIVE transform (maps output points `dst` to input points `src`)."""
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return np.linalg.solve(np.array(A, float), np.array(b, float))


def project(yaw, pitch, f=2600.0, dist=2600.0):
    """Screen positions of the card corners after rotating it in 3D."""
    out = []
    for x, y in [(-CW / 2, -CH / 2), (CW / 2, -CH / 2), (CW / 2, CH / 2), (-CW / 2, CH / 2)]:
        X, Z = x * math.cos(yaw), x * math.sin(yaw)
        Y, Z = y * math.cos(pitch) - Z * math.sin(pitch), y * math.sin(pitch) + Z * math.cos(pitch)
        out.append((W / 2 + f * X / (dist + Z), H / 2 + f * Y / (dist + Z)))
    return out


_RENDER = None


def _worker(i):
    return _RENDER(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--seconds', type=float, default=6)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--sheen', type=float, default=550, help='peak nits of the specular band')
    ap.add_argument('--style', choices=['glow', 'band'], default='glow',
                    help='glow: the art lights up under a soft lobe (default); band: specular stripe + foil')
    ap.add_argument('--glow', type=float, default=2.8, help='extra gain at the centre of the light lobe')
    ap.add_argument('--grain', type=float, default=1.5, help='extra gain on the bright texture grain inside the lobe')
    ap.add_argument('--preview', type=float, nargs='*')
    a = ap.parse_args()
    T = a.seconds

    art = Image.open(a.src).convert('RGB')
    s = max(CW / art.width, CH / art.height)
    art = art.resize((round(art.width * s), round(art.height * s)), Image.LANCZOS)
    l, t = (art.width - CW) // 2, (art.height - CH) // 2
    art = np.asarray(art.crop((l, t, l + CW, t + CH))).astype(np.float32) / 255
    base = (srgb_to_linear(art) @ M709_TO_2020.T).astype(np.float32) * SDR_WHITE
    tex = (base @ LUMA) / SDR_WHITE                                   # texture for the foil
    vv, uu = np.mgrid[0:CH, 0:CW].astype(np.float32)
    u, v = uu / CW, vv / CH
    diag = 0.72 * u + 0.28 * v                                       # sheen runs across this axis
    lt = Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8))
    hp = tex - np.asarray(lt.filter(ImageFilter.GaussianBlur(3))).astype(np.float32) / 255
    grain = np.clip(hp / 0.06, 0, 1)                                  # bright specks of the fuzzy texture
    rng = np.random.default_rng(3)
    n = 2200 if a.style == 'band' else 900
    gx, gy = rng.integers(2, CW - 2, n), rng.integers(2, CH - 2, n)
    gang = rng.uniform(-1, 1, n).astype(np.float32)                   # angle (in units of YAW) where each flashes
    gcol = hsv_rainbow(rng.random(n)).astype(np.float32) * 0.5 + 0.5
    # rounded-corner alpha
    m = Image.new('L', (CW, CH), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, CW - 1, CH - 1], radius=46, fill=255)
    alpha = np.asarray(m.filter(ImageFilter.GaussianBlur(0.8))).astype(np.float32) / 255

    def frame(tt):
        ph = 2 * math.pi * tt / T
        q = math.sin(ph)                                              # -1..1, tilt
        yaw, pitch = YAW * q, PITCH * math.sin(ph + math.pi / 2)
        if a.style == 'band':
            # art + rainbow foil + a narrow specular band + glitter
            c = 0.5 - 0.55 * q
            band = np.exp(-((diag - c) / 0.045) ** 2)
            broad = np.exp(-((diag - c) / 0.13) ** 2)
            foil = hsv_rainbow((diag * 2.2 + q * 0.8 + 0.15 * tex) % 1.0)
            card = base * (1 + 0.15 * q * (u - 0.5))[..., None]
            card = card + foil * (45 * broad * (0.5 + 0.8 * tex))[..., None]
            card = card + (a.sheen * band * (0.55 + 0.45 * tex))[..., None]
            g = np.exp(-((q - gang) / 0.05) ** 2) * 1400
        else:
            # the art itself catches the light: a wide soft lobe multiplies the colours (no white bar),
            # the fuzzy texture's bright grain shimmers inside it, and a little glitter
            cu, cv = 0.5 - 0.85 * q, 0.42 + 0.12 * math.sin(ph + math.pi / 2)
            lobe = np.exp(-(((u - cu) / 0.36) ** 2 + ((v - cv) / 0.7) ** 2))
            gain = 1 + a.glow * lobe + a.grain * lobe * grain
            card = base * gain[..., None]
            card = card + (14 * lobe ** 2)[..., None]                     # faint gloss, barely there
            g = np.exp(-((q - gang) / 0.05) ** 2) * 650 * lobe[gy, gx]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = 1.0 if dx == 0 and dy == 0 else 0.35
                np.add.at(card, (gy + dy, gx + dx), (g * w)[:, None] * gcol)
        # warp onto the screen
        corners = project(yaw, pitch)
        coef = homography([(0, 0), (CW, 0), (CW, CH), (0, CH)], corners)
        warp = lambda ch: np.asarray(Image.fromarray(ch.astype(np.float32), 'F').transform(
            (W, H), Image.PERSPECTIVE, tuple(coef), Image.BILINEAR))
        out_a = warp(alpha)
        rgb = np.dstack([warp(card[..., k]) for k in range(3)])
        # soft shadow on a near-black ground
        sh = Image.fromarray((out_a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(28))
        sh = np.roll(np.asarray(sh).astype(np.float32) / 255, (26, int(-30 * q)), (0, 1))
        bg = np.full((H, W, 3), 0.6, np.float32) * (1 - 0.8 * sh[..., None])
        return bg * (1 - out_a[..., None]) + rgb

    if a.preview is not None:
        for tt in a.preview:
            nits = frame(tt)
            sdr = np.clip((nits @ np.linalg.inv(M709_TO_2020).T) / SDR_WHITE, 0, 1)
            sdr = np.where(sdr <= 0.0031308, sdr * 12.92, 1.055 * sdr ** (1 / 2.4) - 0.055)
            Image.fromarray((sdr * 255).astype(np.uint8)).save(f'{a.out}_{tt:.2f}.png')
            lum = nits @ LUMA
            print(f't={tt:.2f} p99 {np.percentile(lum, 99):.0f} nits  max {lum.max():.0f}  >400 {np.mean(lum > 400) * 100:.2f}%')
        return

    enc = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gbrp16le', '-s', f'{W}x{H}', '-r', str(a.fps),
        '-i', '-', '-vf', 'zscale=min=gbr:m=bt2020nc:r=tv:rin=full,format=yuv420p',
        '-c:v', 'libx264', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-preset', 'slow',
        '-b:v', '10M', '-maxrate', '12M', '-bufsize', '20M',
        '-color_primaries', 'bt2020', '-color_trc', 'arib-std-b67', '-colorspace', 'bt2020nc', '-color_range', 'tv',
        '-x264-params', 'colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc',
        '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
    global _RENDER

    def render(i):
        g = np.round(hlg_from_nits(frame(i / a.fps)) * 65535).astype('<u2')
        return np.ascontiguousarray(g[..., [1, 2, 0]].transpose(2, 0, 1)).tobytes()
    _RENDER = render
    total = int(round(T * a.fps))
    with mp.get_context('fork').Pool(os.cpu_count()) as pool:
        for i, buf in enumerate(pool.imap(_worker, range(total), chunksize=2)):
            enc.stdin.write(buf)
    enc.stdin.close(); enc.wait()
    print(f'{total} frames -> {a.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
