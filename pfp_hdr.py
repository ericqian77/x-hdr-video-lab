"""Make an HDR profile picture for X: a 400x400, 256-colour Rec.2100 PQ PNG.

Format (as reported to keep HDR on X when uploaded from x.com in a desktop browser):
  * 400 x 400, 8-bit indexed PNG, 255 opaque colours + one transparent pixel at (0,0)
  * cICP chunk 9/16/0/1 (BT.2020 primaries, PQ transfer, RGB, full range) + a PQ ICC profile

Grade (SDR -> HDR), all in linear BT.2020 nits, one gain per pixel so hue is kept:
  * base     : SDR level x BASE_GAIN (SDR white = 203 nits)
  * highlights: near-white, low-colour pixels rise toward WHITE_NITS (sparkles, catchlights)
  * pop      : bright, colourful, non-skin pixels (backgrounds, skies) are lifted up to POP x
  * skin guard: warm hues (skin, lips, hair, wood) get no pop and little highlight lift

    python3 pfp_hdr.py in.png out.png [--circle cx cy r] [--pop 7] [--white 2000]
Also writes out_preview.png (SDR side-by-side + nits heat map).
"""
import argparse
import struct
import zlib

import numpy as np
from PIL import Image, ImageFilter

SDR_WHITE = 203.0

# ------------------------------------------------------------------ colour maths
M709_TO_2020 = np.array([[0.627404, 0.329283, 0.043313],
                         [0.069097, 0.919540, 0.011362],
                         [0.016391, 0.088013, 0.895595]])


def srgb_to_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


M1, M2, C1, C2, C3 = 2610 / 16384, 2523 / 4096 * 128, 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq_encode(nits):
    y = np.clip(nits / 10000.0, 0, 1) ** M1
    return ((C1 + C2 * y) / (1 + C3 * y)) ** M2


def pq_decode(e):
    p = np.clip(e, 0, 1) ** (1 / M2)
    return 10000.0 * (np.maximum(p - C1, 0) / (C2 - C3 * p)) ** (1 / M1)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def skin_weight(rgb):
    """1 for warm hues (skin, lips, brown/red hair), fading to 0 for greens, cyans, blues."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60  # degrees
    h = np.where(h > 180, h - 360, h)                     # -180..180, 0 = red
    warm = smoothstep(-40, -10, h) * (1 - smoothstep(55, 85, h))
    return warm


def grade(srgb, base_gain=1.4, white_nits=2000.0, pop=7.0, sat=1.0):
    lin709 = srgb_to_linear(srgb)
    lin = lin709 @ M709_TO_2020.T                         # linear BT.2020, SDR white = 1
    Y = lin @ np.array([0.2627, 0.6780, 0.0593])
    mx = lin.max(-1)
    colourful = (mx - lin.min(-1)) / np.maximum(mx, 1e-6)
    skin = skin_weight(lin709)
    # highlights: near white and low colour -> toward white_nits
    hl = smoothstep(0.80, 1.0, Y) * (1 - smoothstep(0.15, 0.35, colourful)) * (1 - 0.85 * skin)
    hl_gain = 1 + (white_nits / (SDR_WHITE * base_gain) - 1) * hl
    # pop: bright + colourful + not skin
    w = smoothstep(0.25, 0.50, Y) * smoothstep(0.02, 0.10, colourful) * (1 - skin)
    pop_gain = 1 + pop * w
    gain = base_gain * np.maximum(hl_gain, pop_gain)
    # extra colour in the popped areas, so a brightened pale colour stays that colour
    s_ = 1 + (sat - 1) * w
    lin = np.maximum(Y[..., None] + (lin - Y[..., None]) * s_[..., None], 0)
    nits = lin * SDR_WHITE * gain[..., None]
    return nits, Y * SDR_WHITE * gain


# ------------------------------------------------------------------ ICC v4.4 PQ profile (built here)
def _s15(x):
    return struct.pack('>i', int(round(x * 65536)))


def _mluc(text):
    s = text.encode('utf-16-be')
    return b'mluc' + b'\0' * 4 + struct.pack('>II', 1, 12) + b'enUS' + struct.pack('>II', len(s), 28) + s


def _xyz(x, y, z):
    return b'XYZ ' + b'\0' * 4 + _s15(x) + _s15(y) + _s15(z)


def make_pq_icc():
    # BT.2020 RGB -> XYZ (D65), Bradford-adapted to D50 for the PCS.
    xy = {'r': (0.708, 0.292), 'g': (0.170, 0.797), 'b': (0.131, 0.046)}
    wx, wy = 0.3127, 0.3290
    P = np.array([[x / y, 1, (1 - x - y) / y] for x, y in xy.values()]).T
    W = np.array([wx / wy, 1, (1 - wx - wy) / wy])
    M = P * np.linalg.solve(P, W)
    brad = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
    d50 = np.array([0.9642, 1.0, 0.8249])
    ratio = (brad @ d50) / (brad @ W)
    chad = np.linalg.inv(brad) @ np.diag(ratio) @ brad
    M50 = chad @ M
    # TRC: PQ code -> relative linear with 203 nits = 1.0 (clipped), so colour-managed
    # SDR viewers see the SDR-level picture and clip the HDR highlights.
    n = 4096
    lin = np.clip(pq_decode(np.linspace(0, 1, n)) / SDR_WHITE, 0, 1)
    curv = b'curv' + b'\0' * 4 + struct.pack('>I', n) + (np.round(lin * 65535).astype('>u2')).tobytes()
    tags = [
        (b'desc', _mluc('Rec.2100 PQ (x-hdr-video-lab)')),
        (b'cprt', _mluc('No copyright, use freely')),
        (b'wtpt', _xyz(*d50)),
        (b'chad', b'sf32' + b'\0' * 4 + b''.join(_s15(v) for v in chad.flatten())),
        (b'rXYZ', _xyz(*M50[:, 0])), (b'gXYZ', _xyz(*M50[:, 1])), (b'bXYZ', _xyz(*M50[:, 2])),
        (b'rTRC', curv), (b'gTRC', curv), (b'bTRC', curv),
        (b'cicp', b'cicp' + b'\0' * 4 + bytes([9, 16, 0, 1])),
        (b'lumi', _xyz(0, 10000, 0)),
    ]
    off = 128 + 4 + 12 * len(tags)
    table, data, placed = b'', b'', {}
    for sig, body in tags:
        if body in placed:                      # share identical bodies (the three TRCs)
            o = placed[body]
        else:
            while (off + len(data)) % 4:
                data += b'\0'
            o = off + len(data); placed[body] = o; data += body
        table += sig + struct.pack('>II', o, len(body))
    total = off + len(data)
    hdr = (struct.pack('>I', total) + b'\0' * 4 + bytes([4, 0x40, 0, 0]) + b'mntrRGB XYZ '
           + struct.pack('>6H', 2026, 1, 1, 0, 0, 0) + b'acsp' + b'\0' * 4 + b'\0' * 4 + b'\0' * 8 + b'\0' * 8
           + b'\0' * 4 + _s15(d50[0]) + _s15(d50[1]) + _s15(d50[2]) + b'\0' * 4 + b'\0' * 16 + b'\0' * 28)
    assert len(hdr) == 128
    return hdr + struct.pack('>I', len(tags)) + table + data


# ------------------------------------------------------------------ PNG writer
def _chunk(t, d):
    return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)


def write_indexed_pq_png(path, idx, palette, transparent_index):
    h, w = idx.shape
    trns = bytes([255] * transparent_index + [0])
    raw = b''.join(b'\0' + idx[y].astype(np.uint8).tobytes() for y in range(h))
    icc = b'Rec2100 PQ\0\0' + zlib.compress(make_pq_icc(), 9)
    png = (b'\x89PNG\r\n\x1a\n'
           + _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 3, 0, 0, 0))
           + _chunk(b'cICP', bytes([9, 16, 0, 1]))
           + _chunk(b'iCCP', icc)
           + _chunk(b'PLTE', palette.astype(np.uint8).tobytes())
           + _chunk(b'tRNS', trns)
           + _chunk(b'IDAT', zlib.compress(raw, 9))
           + _chunk(b'IEND', b''))
    open(path, 'wb').write(png)


def write_rgb16_pq_png(path, codes16):
    h, w, _ = codes16.shape
    raw = b''.join(b'\0' + codes16[y].astype('>u2').tobytes() for y in range(h))
    icc = b'Rec2100 PQ\0\0' + zlib.compress(make_pq_icc(), 9)
    png = (b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 16, 2, 0, 0, 0))
           + _chunk(b'cICP', bytes([9, 16, 0, 1])) + _chunk(b'iCCP', icc)
           + _chunk(b'IDAT', zlib.compress(raw, 9)) + _chunk(b'IEND', b''))
    open(path, 'wb').write(png)


# ------------------------------------------------------------------ main
def circle_crop(img, cx, cy, r):
    """Square crop around a circle; pixels outside the circle repeat the rim radially."""
    a = np.asarray(img.convert('RGB')).astype(np.float32)
    side = int(2 * r)
    ys, xs = np.mgrid[0:side, 0:side].astype(np.float32) + 0.5
    dx, dy = xs - side / 2, ys - side / 2
    d = np.hypot(dx, dy)
    k = np.minimum(1, (r - 4) / np.maximum(d, 1e-6))
    sx = np.clip(cx + dx * k, 0, a.shape[1] - 1).astype(int)
    sy = np.clip(cy + dy * k, 0, a.shape[0] - 1).astype(int)
    return Image.fromarray(a[sy, sx].astype(np.uint8))


def background_mask(srgb, tol=0.12):
    """Soft mask (0..1) of the flat background: pixels near the border colour that connect to the border."""
    rim = np.concatenate([srgb[0], srgb[-1], srgb[:, 0], srgb[:, -1]])
    ref = np.median(rim, 0)
    near = np.linalg.norm(srgb - ref, axis=-1) < tol
    seed = np.zeros_like(near); seed[0] = seed[-1] = seed[:, 0] = seed[:, -1] = True
    m = seed & near
    while True:                                   # grow inside `near`, 4-connected
        g = m.copy(); g[1:] |= m[:-1]; g[:-1] |= m[1:]; g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
        g &= near
        if (g == m).all():
            break
        m = g
    img = Image.fromarray((m * 255).astype(np.uint8))
    img = img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))   # close sparkle holes
    img = img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.0))  # cover the old fringe, feather
    return np.asarray(img).astype(np.float32) / 255, ref


def recolour_background(srgb, mask, ref, colour, peak_nits, white_nits):
    """Background nits in a new saturated colour; keeps the old shading and white sparkles."""
    lin = srgb_to_linear(srgb) @ M709_TO_2020.T
    w = np.array([0.2627, 0.6780, 0.0593])
    Y = lin @ w
    Yref = (srgb_to_linear(ref) @ M709_TO_2020.T) @ w
    shade = np.clip(Y / Yref, 0.6, 1.2)
    sparkle = smoothstep(0.04, 0.25, Y - Yref)
    c = srgb_to_linear(np.array(colour) / 255) @ M709_TO_2020.T
    c = c / c.max() * peak_nits                   # brightest channel of the new colour at peak_nits
    bg = c * shade[..., None]
    bg = bg + (np.full(3, white_nits) - bg) * sparkle[..., None]
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--circle', type=float, nargs=3, metavar=('CX', 'CY', 'R'))
    ap.add_argument('--base', type=float, default=1.4)
    ap.add_argument('--white', type=float, default=2000)
    ap.add_argument('--pop', type=float, default=7.0)
    ap.add_argument('--sat', type=float, default=1.0, help='colour boost in popped areas')
    ap.add_argument('--bg', help='replace the flat background with this colour, e.g. 00c878')
    ap.add_argument('--bg-nits', type=float, default=800, help='brightest channel of the new background')
    a = ap.parse_args()

    img = Image.open(a.src).convert('RGB')
    if a.circle:
        img = circle_crop(img, *a.circle)
    else:
        s = min(img.size); img = img.crop(((img.width - s) // 2, (img.height - s) // 2,
                                           (img.width + s) // 2, (img.height + s) // 2))
    # resample in linear light
    lin = srgb_to_linear(np.asarray(img).astype(np.float32) / 255)
    lin = np.stack([np.asarray(Image.fromarray(lin[..., c]).resize((400, 400), Image.BOX if img.width >= 400 else Image.BICUBIC))
                    for c in range(3)], -1)
    srgb = linear_to_srgb(np.clip(lin, 0, 1))

    nits, lum = grade(srgb, a.base, a.white, a.pop, a.sat)
    if a.bg:
        mask, ref = background_mask(srgb)
        colour = [int(a.bg.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)]
        bg = recolour_background(srgb, mask, ref, colour, a.bg_nits, a.white)
        nits = nits * (1 - mask[..., None]) + bg * mask[..., None]
        sdr_bg = linear_to_srgb(np.clip((bg @ np.linalg.inv(M709_TO_2020).T) / a.bg_nits, 0, 1))
        img = Image.fromarray(np.round(255 * (srgb * (1 - mask[..., None]) + sdr_bg * mask[..., None])).astype(np.uint8))
    pq = pq_encode(nits)
    codes8 = np.round(pq * 255).astype(np.uint8)

    # 255-colour palette in PQ code space with Floyd-Steinberg dithering
    q = Image.fromarray(codes8).quantize(colors=255, method=Image.Quantize.MEDIANCUT, kmeans=4,
                                         dither=Image.Dither.FLOYDSTEINBERG)
    idx = np.asarray(q).copy()
    pal = np.zeros((256, 3), np.uint8)
    pl = np.array(q.getpalette()[:255 * 3], np.uint8).reshape(-1, 3)
    pal[:len(pl)] = pl
    pal[255] = pal[idx[0, 0]]
    idx[0, 0] = 255                       # the one transparent pixel
    write_indexed_pq_png(a.out, idx, pal, 255)
    base = a.out.rsplit('.', 1)[0]
    write_rgb16_pq_png(base + '_16bit.png', np.round(pq * 65535).astype(np.uint16))

    # diagnostics
    q_nits = pq_decode(pal[idx].astype(np.float32) / 255)
    ql = q_nits @ np.array([0.2627, 0.6780, 0.0593])
    print(f'luminance nits: median {np.median(ql):.0f}  p90 {np.percentile(ql, 90):.0f}  max {ql.max():.0f}')
    for name, (y0, y1, x0, x1) in {'background': (20, 80, 300, 380), 'face': (180, 230, 150, 250),
                                   'shirt': (340, 390, 60, 160), 'hair': (40, 70, 160, 240)}.items():
        print(f'  {name:10s} median {np.median(ql[y0:y1, x0:x1]):.0f} nits')
    print(f'pixels >400 nits {np.mean(ql > 400) * 100:.1f}%  >1000 nits {np.mean(ql > 1000) * 100:.1f}%')
    heat = np.clip(np.log10(np.maximum(ql, 1)) / np.log10(3000), 0, 1)
    cmap = np.stack([np.interp(heat, [0, .5, .75, 1], c) for c in
                     ([0, 40, 255, 255], [0, 40, 140, 255], [60, 200, 0, 220])], -1)
    sdr = np.asarray(img.resize((400, 400), Image.LANCZOS))
    Image.fromarray(np.concatenate([sdr, cmap.astype(np.uint8)], 1)).save(base + '_preview.png')
    Image.fromarray(sdr).save(base + '_sdr.png')


if __name__ == '__main__':
    main()
