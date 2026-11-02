"""Colour-targeted HDR grade for video: lift saturated colours in a hue band, leave pale areas alone.

A global curve lifts every bright pixel, so a pale sky clips to white before the subject glows.
Here one gain per pixel (hue kept) is computed from its colour:
  * base    : SDR level x --base (SDR white = 203 nits)
  * pop     : pixels inside the hue band (degrees, 0 = red, 60 = yellow, 120 = green, 240 = blue),
              colourful and not dark, get up to 1 + --pop
  * whites  : near-white, low-colour pixels rise toward --white nits (small highlights)
Output: HLG BT.2020 (1000-nit reference), H.264 High 8-bit, tagged in container and stream,
long side scaled to 1920, audio copied as AAC.

    python3 grade_pop.py in.mp4 out.mp4 [--hue 20 75] [--pop 3] [--base 1.1] [--white 600] [--level 0.05 0.25]
The hue band may wrap past 360, e.g. --hue 270 70 covers purple, magenta, red, orange and yellow.
"""
import argparse
import json
import subprocess
import sys

import numpy as np

from pfp_hdr import M709_TO_2020, SDR_WHITE, smoothstep, srgb_to_linear

LUMA = np.array([0.2627, 0.6780, 0.0593])


def bt709_to_linear(v):
    return np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45))


def hue_deg(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h % 360


def band(h, lo, hi, soft=15):
    """1 inside [lo, hi] degrees, fading out over `soft` degrees (handles wrap-around)."""
    width = (hi - lo) % 360
    off = (h - lo) % 360
    inside = off <= width
    dist = np.where(inside, 0, np.minimum(off - width, 360 - off))
    return 1 - smoothstep(0, soft, dist)


def grade_frame(code709, a):
    lin709 = bt709_to_linear(code709)
    lin = lin709 @ M709_TO_2020.T
    Y = lin @ LUMA
    mx = lin.max(-1)
    colourful = (mx - lin.min(-1)) / np.maximum(mx, 1e-6)
    w = band(hue_deg(lin709), *a.hue) * smoothstep(0.35, 0.6, colourful) * smoothstep(*a.level, mx)
    hl = smoothstep(0.8, 1.0, Y) * (1 - smoothstep(0.1, 0.3, colourful))
    gain = a.base * np.maximum(1 + a.pop * w, 1 + (a.white / (SDR_WHITE * a.base) - 1) * hl)
    return lin * SDR_WHITE * gain[..., None], w


def hlg_from_nits(nits, peak=1000.0):
    """Display light (nits, BT.2020) -> HLG signal for a `peak`-nit reference display (BT.2100 inverse OOTF + OETF)."""
    gamma = 1.2
    Fd = np.clip(nits / peak, 0, 1)
    Yd = np.maximum(Fd @ LUMA, 1e-9)
    E = Fd * (Yd ** ((1 - gamma) / gamma))[..., None]
    E = np.clip(E, 0, 1)
    a, b = 0.17883277, 1 - 4 * 0.17883277
    c = 0.5 - a * np.log(4 * a)
    return np.where(E <= 1 / 12, np.sqrt(3 * E), a * np.log(np.maximum(12 * E - b, 1e-9)) + c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--hue', type=float, nargs=2, default=[20, 75], metavar=('LO', 'HI'))
    ap.add_argument('--pop', type=float, default=3.0)
    ap.add_argument('--base', type=float, default=1.1)
    ap.add_argument('--white', type=float, default=600)
    ap.add_argument('--level', type=float, nargs=2, default=[0.05, 0.25], metavar=('LO', 'HI'),
                    help='brightness (max channel, linear) where the lift fades in')
    ap.add_argument('--mask', help='also write an SDR preview of the pop mask to this mp4')
    a = ap.parse_args()

    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                      'stream=width,height,r_frame_rate', '-of', 'json', a.src],
                                     capture_output=True, text=True).stdout)['streams'][0]
    w, h, fps = info['width'], info['height'], info['r_frame_rate']
    ow, oh = (1920, round(1920 * h / w / 2) * 2) if w >= h else (round(1920 * w / h / 2) * 2, 1920)

    dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', a.src, '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-'],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'gbrp16le', '-s', f'{w}x{h}', '-r', fps, '-i', '-',
        '-i', a.src, '-map', '0:v', '-map', '1:a?',
        '-vf', f'scale={ow}:{oh}:flags=lanczos,zscale=min=gbr:m=bt2020nc:r=tv:rin=full,format=yuv420p',
        '-c:v', 'libx264', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-b:v', '10M', '-maxrate', '12M', '-bufsize', '20M',
        '-color_primaries', 'bt2020', '-color_trc', 'arib-std-b67', '-colorspace', 'bt2020nc', '-color_range', 'tv',
        '-x264-params', 'colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc',
        '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
    menc = None
    if a.mask:
        menc = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                                 '-r', fps, '-i', '-', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', a.mask],
                                stdin=subprocess.PIPE)
    n, frame_bytes = 0, w * h * 6
    stats = []
    while True:
        buf = dec.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        code = np.frombuffer(buf, '<u2').reshape(h, w, 3).astype(np.float32) / 65535
        nits, wmask = grade_frame(code, a)
        sig = hlg_from_nits(nits)
        g = np.round(sig * 65535).astype('<u2')
        enc.stdin.write(np.ascontiguousarray(g[..., [1, 2, 0]].transpose(2, 0, 1)).tobytes())   # planar G, B, R
        if menc:
            menc.stdin.write((np.dstack([wmask] * 3) * 255).astype(np.uint8).tobytes())
        if n % 10 == 0:
            stats.append(nits @ LUMA)
        n += 1
    enc.stdin.close(); enc.wait()
    if menc:
        menc.stdin.close(); menc.wait()
    s = np.concatenate([x.ravel() for x in stats])
    print(f'{n} frames  luminance nits: median {np.median(s):.0f}  p90 {np.percentile(s, 90):.0f}  '
          f'p99 {np.percentile(s, 99):.0f}  >400 nits {np.mean(s > 400) * 100:.1f}%', file=sys.stderr)


if __name__ == '__main__':
    main()
