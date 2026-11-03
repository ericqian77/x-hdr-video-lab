"""Turn still images into a photographic-looking motion clip (SDR master).

Every output pixel is resampled from the source still; nothing is drawn or
generated. Motion comes from:
  * a slow camera dolly with depth parallax (Depth Anything V2 depth map),
  * water: travelling ripples on the reflection / seabed, with caustic
    focusing derived from the wave height field,
  * subtle subject motion per scene (rotation, spin, sway),
  * the same subject motion mirrored into its reflection.

Usage:
  python3 animate_stills.py --images DIR --depth-model da2s.onnx --out master.mkv
Then grade to HLG with grade_hlg.sh.
"""
import argparse
import math
import os
import subprocess
from multiprocessing import Pool

import cv2
import numpy as np

W_OUT, H_OUT, FPS = 1920, 1080, 24

# ---------------------------------------------------------------- scenes
# Coordinates are in source pixels (images are 1536x1024).
SCENES = [
    dict(  # 1: petal nautilus over turquoise shallows
        file="1.webp", dur=7.0, yh=700, refl_c=0.66,
        cam=dict(c0=(768, 470), c1=(700, 455), z0=1.00, z1=1.09, dolly=0.06, truck=-14),
        water=dict(amp=2.2, lam=(0.06, 0.45), speed=1.0, caustic=0.08, glint=0.45, n=14),
        subject=dict(kind="spiral", center=(456, 354), r_in=330, r_out=470, angle=-5.0,
                     ribbon_x0=640, ribbon_amp=7.0, ribbon_lam=520, ribbon_period=4.5),
        boat=dict(box=(1040, 760, 1290, 865), bob=2.2, period=3.4),
        clouds=0.8,
        keep_out=[],
    ),
    dict(  # 2: ginkgo lantern sphere over the mirror lake
        file="2.webp", dur=7.0, yh=710, refl_c=1.0,
        cam=dict(c0=(768, 450), c1=(768, 480), z0=1.00, z1=1.08, dolly=0.05, truck=12),
        water=dict(amp=0.9, lam=(0.05, 0.25), speed=0.55, caustic=0.0, glint=0.0, n=12),
        subject=dict(kind="sphere", center=(768, -25), radius=632, angle=6.0, glow=0.10,
                     glow_period=3.5),
        clouds=1.0,
        keep_out=[  # bridge / walkway, woman
            np.array([(0, 731), (300, 726), (600, 716), (690, 700), (840, 700), (940, 712),
                      (1100, 724), (1300, 738), (1536, 750), (1536, 795), (1300, 774),
                      (1100, 760), (940, 754), (830, 752), (700, 750), (600, 748),
                      (300, 752), (0, 756)], np.int32),
            np.array([(965, 655), (1015, 655), (1015, 740), (965, 740)], np.int32),
        ],
    ),
    dict(  # 3: lace hibiscus over the salt flat
        file="3.webp", dur=7.0, yh=629, refl_c=1.0,
        cam=dict(c0=(860, 420), c1=(768, 432), z0=1.12, z1=1.00, dolly=-0.05, truck=16),
        water=dict(amp=0.8, lam=(0.05, 0.25), speed=0.5, caustic=0.0, glint=0.0, n=12),
        subject=dict(kind="flower", pivot=(985, 330), base=(1330, 636), sway=1.3,
                     sway_period=5.5, flutter=3.2, x_min=420),
        dress=dict(box=(262, 560, 372, 684), amp=1.6),
        clouds=0.8,
        keep_out=[
            np.array([(0, 655), (300, 660), (700, 668), (520, 688), (220, 708), (0, 724)], np.int32),
            np.array([(262, 545), (372, 545), (372, 684), (262, 684)], np.int32),
        ],
    ),
]
XFADE = 1.0  # seconds of cross-dissolve between scenes


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


# ---------------------------------------------------------------- depth
def estimate_depth(img_rgb, model_path):
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    H, W = img_rgb.shape[:2]
    x = cv2.resize(img_rgb.astype(np.float32) / 255, (770, 518), interpolation=cv2.INTER_CUBIC)
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    d = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    return ((d - d.min()) / (d.max() - d.min())).astype(np.float32)


# ---------------------------------------------------------------- per-scene prep
class Scene:
    def __init__(self, cfg, img_dir, model_path, seed):
        self.cfg = cfg
        bgr = cv2.imread(os.path.join(img_dir, cfg["file"]), cv2.IMREAD_COLOR)
        self.img = bgr.astype(np.float32) / 255.0
        H, W = self.img.shape[:2]
        self.W, self.H = W, H
        yh, c = cfg["yh"], cfg["refl_c"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = estimate_depth(rgb, model_path)
        # Reflections live at the mirrored object's depth, not at the water plane.
        ys = np.arange(H, dtype=np.float32)
        src_rows = np.clip(yh - (ys - yh) / c, 0, H - 1)
        mirrored = cv2.remap(depth, np.tile(np.arange(W, dtype=np.float32), (H, 1)),
                             np.repeat(src_rows[:, None], W, 1), cv2.INTER_LINEAR)
        wmask = smoothstep(yh + 1, yh + 6, ys)[:, None] * np.ones((1, W), np.float32)
        self.depth = cv2.GaussianBlur(depth * (1 - wmask) + mirrored * wmask, (0, 0), 3)

        # water mask minus solid things standing in it
        keep = np.zeros((H, W), np.float32)
        for poly in cfg["keep_out"]:
            cv2.fillPoly(keep, [poly], 1.0)
        keep = cv2.GaussianBlur(keep, (0, 0), 3)
        self.water = (wmask * (1 - keep)).astype(np.float32)
        self.keep = keep

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat, val = hsv[..., 1] / 255, hsv[..., 2] / 255
        sky = (ys[:, None] < yh - 2).astype(np.float32)
        subj = (depth > 0.42).astype(np.float32) * sky
        subj = cv2.GaussianBlur(cv2.dilate(subj, np.ones((15, 15), np.uint8)), (0, 0), 6)
        self.subject = subj
        cloud = ((sat < 0.30) & (val > 0.62)).astype(np.float32) * sky * (1 - subj)
        self.cloud = cv2.GaussianBlur(cloud, (0, 0), 4)
        # cyan seabed (for caustics) in scene 1
        hue = hsv[..., 0] * 2
        self.seabed = (smoothstep(150, 170, hue) * (1 - smoothstep(195, 215, hue)) *
                       smoothstep(0.35, 0.55, sat)) * self.water
        self.seabed = cv2.GaussianBlur(self.seabed.astype(np.float32), (0, 0), 3)
        # luminous interior of the lantern sphere (scene 2)
        self.glow = (smoothstep(15, 25, hue) * (1 - smoothstep(48, 60, hue)) *
                     smoothstep(0.80, 0.95, val)) * (subj + self.water)
        self.glow = cv2.GaussianBlur(self.glow.astype(np.float32), (0, 0), 2)

        rng = np.random.default_rng(seed)
        n = cfg["water"]["n"]
        lo, hi = cfg["water"]["lam"]
        lam = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        ang = rng.normal(0, 0.9, n) + math.pi / 2  # mostly travelling toward camera
        k = 2 * math.pi / lam
        self.wk = np.stack([k * np.cos(ang), k * np.sin(ang)], 1).astype(np.float32)
        self.wlam = lam.astype(np.float32)
        self.wom = (np.sqrt(9.81 * k * 0.02) * cfg["water"]["speed"]).astype(np.float32)
        self.wph = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
        self.wamp = (lam / lam.max()) ** 0.7  # longer waves taller
        self.wnorm, self.lnorm = 1.0, 1.0
        gy, gx = np.mgrid[yh + 2:H:3, 0:W:3].astype(np.float32)
        _, hz, lap, _ = self.water_field(gx, gy, 0.0)
        self.wnorm, self.lnorm = float(hz.std()), float(lap.std())
        # generic smooth noise components (clouds, flutter)
        self.nk = rng.normal(0, 1, (6, 2)).astype(np.float32)
        self.nph = rng.uniform(0, 2 * math.pi, (6, 2)).astype(np.float32)
        self.nom = rng.uniform(0.6, 1.4, 6).astype(np.float32)

    # smooth 2D vector noise, wavelength ~lam px, unit-ish amplitude
    def vnoise(self, x, y, t, lam, period):
        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        for i in range(6):
            kx, ky = self.nk[i] * (2 * math.pi / lam)
            ph = kx * x + ky * y + self.nom[i] * 2 * math.pi * t / period
            dx += np.sin(ph + self.nph[i, 0])
            dy += np.cos(ph * 0.93 + self.nph[i, 1])
        return dx / 2.4, dy / 2.4

    # ------------------------------------------------------------ fields
    def subject_disp(self, x, y, t, T):
        """Displacement (content motion) of the floating subject at source point."""
        s = self.cfg["subject"]
        p = t / T
        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        if s["kind"] == "spiral":
            cx, cy = s["center"]
            rx, ry = x - cx, y - cy
            r = np.sqrt(rx * rx + ry * ry)
            th = math.radians(s["angle"]) * ease(p) * (1 - smoothstep(s["r_in"], s["r_out"], r))
            th = th * (1 - smoothstep(s["ribbon_x0"] - 60, s["ribbon_x0"] + 60, x) * smoothstep(-80, 80, ry))
            c, sn = np.cos(th), np.sin(th)
            dx += rx * c - ry * sn - rx
            dy += rx * sn + ry * c - ry
            rib = smoothstep(s["ribbon_x0"], s["ribbon_x0"] + 300, x)
            a = s["ribbon_amp"] * rib * (0.4 + 0.6 * (x - s["ribbon_x0"]) / (1536 - s["ribbon_x0"]))
            ph = 2 * math.pi * (x / s["ribbon_lam"] - t / s["ribbon_period"])
            dy += a * np.sin(ph)
            dx += 0.35 * a * np.cos(ph * 1.3 + 0.7)
        elif s["kind"] == "sphere":
            cx, cy = s["center"]
            R = s["radius"]
            u = (x - cx) / R
            v = (y - cy) / R
            rr = np.clip(1 - u * u - v * v, 0, None)
            z = np.sqrt(rr)
            a = math.radians(s["angle"]) * (p - 0.5) * 2
            # content at (u,v,z) moves to rotated position about the vertical axis
            u2 = u * math.cos(a) + z * math.sin(a)
            inside = smoothstep(0.0, 0.02, rr)
            dx += (u2 - u) * R * inside
        elif s["kind"] == "flower":
            px, py = s["pivot"]
            bx, by = s["base"]
            L = math.hypot(px - bx, py - by)
            d_base = np.sqrt((x - bx) ** 2 + (y - by) ** 2)
            w = np.clip(d_base / L, 0, 1.15) ** 1.6
            w *= smoothstep(s["x_min"] - 80, s["x_min"] + 40, x)
            ang = math.radians(s["sway"]) * math.sin(2 * math.pi * t / s["sway_period"])
            rx, ry = x - bx, y - by  # bend about stem base
            c, sn = math.cos(ang * 1.0), math.sin(ang * 1.0)
            dx += (rx * c - ry * sn - rx) * w
            dy += (rx * sn + ry * c - ry) * w
            # petal flutter grows away from the flower centre
            rp = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            fw = smoothstep(80, 420, rp) * smoothstep(s["x_min"] - 80, s["x_min"] + 40, x)
            nx, ny = self.vnoise(x, y, t, 260.0, 3.2)
            dx += s["flutter"] * fw * nx
            dy += s["flutter"] * fw * ny
        return dx, dy

    def water_field(self, x, y, t):
        cfg, yh, H = self.cfg["water"], self.cfg["yh"], self.H
        depth_px = np.maximum(y - yh, 0.5)
        persp = depth_px / (H - yh)
        Zp = 1.0 / persp
        Xp = (x - self.W / 2) / (H - yh) * Zp
        hx = np.zeros_like(x)
        hz = np.zeros_like(x)
        lap = np.zeros_like(x)
        for i in range(len(self.wlam)):
            lam_img = self.wlam[i] * (H - yh) * persp * persp  # foreshortened
            fade = smoothstep(3.0, 10.0, lam_img) * self.wamp[i]
            kx, kz = self.wk[i]
            ph = kx * Xp + kz * Zp - self.wom[i] * t + self.wph[i]
            c = np.cos(ph) * fade
            hx += kx * c * self.wlam[i]
            hz += kz * c * self.wlam[i]
            lap -= np.sin(ph) * fade
        return hx / self.wnorm, hz / self.wnorm, lap / self.lnorm, persp

    # ------------------------------------------------------------ render
    def render(self, t):
        cfg = self.cfg
        T = cfg["dur"]
        p = ease(t / T)
        cam = cfg["cam"]
        W, H, yh, c_ref = self.W, self.H, cfg["yh"], cfg["refl_c"]

        # camera: output pixel -> source point
        z = cam["z0"] + (cam["z1"] - cam["z0"]) * p
        cx = cam["c0"][0] + (cam["c1"][0] - cam["c0"][0]) * p
        cy = cam["c0"][1] + (cam["c1"][1] - cam["c0"][1]) * p
        S = (W_OUT / W) * z
        qx, qy = np.meshgrid(np.arange(W_OUT, dtype=np.float32), np.arange(H_OUT, dtype=np.float32))
        x0 = cx + (qx - W_OUT / 2) / S
        y0 = cy + (qy - H_OUT / 2) / S

        d = cv2.remap(self.depth, x0, y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # dolly: expansion about the focus of expansion on the horizon
        fx, fy = cx, yh
        a = cam["dolly"] * p
        x1 = fx + (x0 - fx) * (1 - a * d)
        y1 = fy + (y0 - fy) * (1 - a * d)
        x1 += cam["truck"] * (p - 0.5) * (d - 0.3)

        def samp(m):
            return cv2.remap(m, x0, y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        subj, water, cloud, keep = samp(self.subject), samp(self.water), samp(self.cloud), samp(self.keep)

        # subject motion, mirrored into the reflection
        below = y0 > yh
        ym = np.where(below, yh - (y0 - yh) / c_ref, y0)
        sdx, sdy = self.subject_disp(x0, ym, t, T)
        sdy = np.where(below, -sdy * c_ref, sdy)
        msub = cv2.remap(self.subject, x0, ym, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        msub = np.where(below, msub * water, msub)
        if cfg["subject"]["kind"] == "sphere":
            msub = np.where(below, water, 1.0) * (1 - keep)
        dx = sdx * msub
        dy = sdy * msub

        # water ripples
        hx, hz, lap, persp = self.water_field(x0, y0, t)
        wa = cfg["water"]["amp"]
        wscale = water * np.clip(persp, 0, 1) ** 0.8
        dx += wa * 0.45 * hx * wscale
        dy += wa * 1.0 * hz * wscale

        # clouds: very slow billow
        if cfg.get("clouds"):
            nx, ny = self.vnoise(x0, ym, t, 140.0, 9.0)
            cm = np.where(below, cv2.remap(self.cloud, x0, ym, cv2.INTER_LINEAR) * water, cloud)
            dx += cfg["clouds"] * nx * cm
            dy += cfg["clouds"] * 0.6 * ny * cm

        if "boat" in cfg:
            b = cfg["boat"]
            x_a, y_a, x_b, y_b = b["box"]
            bm = (smoothstep(x_a - 25, x_a + 5, x0) * (1 - smoothstep(x_b - 5, x_b + 25, x0)) *
                  smoothstep(y_a - 20, y_a + 5, y0) * (1 - smoothstep(y_b - 5, y_b + 20, y0)))
            ph = 2 * math.pi * t / b["period"]
            roll = 0.012 * math.sin(ph + 1.1)
            bob = b["bob"] * math.sin(ph)
            bcx = (x_a + x_b) / 2
            dy += bm * (bob + roll * (x0 - bcx))
            dx += bm * 0.8 * math.sin(ph * 0.5)

        if "dress" in cfg:
            b = cfg["dress"]
            x_a, y_a, x_b, y_b = b["box"]
            hem = smoothstep(y_a, y_b, ym) * smoothstep(x_a - 10, x_a + 30, x0) * (1 - smoothstep(x_b - 50, x_b, x0))
            hem *= smoothstep(x_a - 30, x_a, x0) * (1 - smoothstep(y_b, y_b + 10, ym))
            nx, ny = self.vnoise(x0, ym, t, 60.0, 1.8)
            dx += b["amp"] * nx * hem
            dy += b["amp"] * 0.5 * np.where(below, -ny, ny) * hem

        mx = x1 - dx
        my = y1 - dy
        out = cv2.remap(self.img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

        # caustic focusing on the visible seabed (physically: brightness ~ -laplacian h)
        cs = cfg["water"]["caustic"]
        if cs:
            sb = samp(self.seabed)
            out *= (1 + cs * sb * np.clip(lap, -1.5, 1.5))[..., None]
        # sun glints: sharp specular hits on the steepest wave facets
        gs = cfg["water"]["glint"]
        if gs:
            slope = np.sqrt(hx * hx + hz * hz)
            g = smoothstep(2.7, 3.6, slope) * water * smoothstep(0.02, 0.15, persp)
            g *= (1 - smoothstep(0.55, 0.9, persp))
            g = cv2.GaussianBlur(g, (0, 0), 0.8)
            out += (gs * g)[..., None] * np.array([1.0, 0.98, 0.94], np.float32)
        if cfg["subject"]["kind"] == "sphere":
            s = cfg["subject"]
            gl = cv2.remap(self.glow, mx, my, cv2.INTER_LINEAR)
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * t / s["glow_period"] - 1.2)
            out *= (1 + s["glow"] * pulse * gl)[..., None]

        # gentle sharpening to offset the resample, plus fine grain against banding
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = out + 0.35 * (out - blur)
        return out


def render_frame(args):
    idx, = args
    t = idx / FPS
    seg = []
    start = 0.0
    for i, sc in enumerate(SCENES):
        seg.append((start, start + sc["dur"], i))
        start += sc["dur"] - XFADE
    layers = []
    for s0, s1, i in seg:
        if s0 <= t < s1:
            w = 1.0
            if i > 0 and t < s0 + XFADE:
                w = smoothstep(0, 1, (t - s0) / XFADE)
            layers.append((i, t - s0, w))
    frame = None
    for i, lt, w in layers:
        f = G_SCENES[i].render(lt)
        frame = f if frame is None else frame * (1 - w) + f * w
    rng = np.random.default_rng(idx)
    frame += rng.normal(0, 0.6 / 255, frame.shape[:2])[..., None].astype(np.float32)
    return np.clip(frame[..., ::-1] * 65535 + 0.5, 0, 65535).astype(np.uint16).tobytes()


G_SCENES = []


def init_worker(img_dir, model):
    global G_SCENES
    cv2.setNumThreads(1)
    G_SCENES = [Scene(c, img_dir, model, seed=11 + i) for i, c in enumerate(SCENES)]


def total_frames():
    return int(round((sum(s["dur"] for s in SCENES) - XFADE * (len(SCENES) - 1)) * FPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--depth-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="", help="comma list of frame indices to dump as PNG instead")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()

    if a.frames:
        init_worker(a.images, a.depth_model)
        for f in a.frames.split(","):
            buf = render_frame((int(f),))
            im = np.frombuffer(buf, np.uint16).reshape(H_OUT, W_OUT, 3)[..., ::-1]
            cv2.imwrite(f"{a.out}_{int(f):04d}.png", im)
        return

    n = total_frames()
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le",
        "-s", f"{W_OUT}x{H_OUT}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "8", "-pix_fmt", "yuv444p10le",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        a.out], stdin=subprocess.PIPE)
    with Pool(a.jobs, initializer=init_worker, initargs=(a.images, a.depth_model)) as pool:
        for i, buf in enumerate(pool.imap(render_frame, [(k,) for k in range(n)], chunksize=2)):
            ff.stdin.write(buf)
            if i % 24 == 0:
                print(f"frame {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()
