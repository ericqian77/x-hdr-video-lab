# x-hdr-video-lab

Turn an SDR video (for example an AI-generated clip) into an HLG HDR file that plays as HDR in the X (Twitter) iPhone app.

These notes come from our own tests, not from any official X spec.

## What worked
- H.264 High, 8-bit, HLG (`arib-std-b67`), BT.2020 primaries and matrix, with the tags written both in the container and in the x264 stream.
- Upload the finished file from a desktop browser at x.com. Uploading from the iPhone photo album dropped HDR in our tests.
- Check on an HDR iPhone in the X app.

## Usage
```bash
# default curve
bash grade_hlg.sh in.mp4 out.mp4
# brighter preset for fast scenes with small white highlights
PRE="eq=brightness=0.03," CURVE="0/0 0.32/0.28 0.55/0.60 0.78/0.82 1/1" bash grade_hlg.sh in.mp4 out.mp4
# check tags and brightness
ffprobe -v error -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries -of default=nw=1 out.mp4
python3 ystat.py out.mp4
```
The output keeps the input's aspect ratio, with the long side at 1920 (so 1920x1080 or 1080x1920). Needs ffmpeg built with zscale (libzimg), and Python with numpy.

Curve points act on linear light, not SDR code values: an SDR code value of 0.5 is about 0.26 in linear light, and SDR white maps to the HLG peak. Untagged inputs are treated as BT.709 limited range.

A result that looked good: median luma about 0.54, about 24% of pixels above 0.75. A median under 0.4 looked too dark. Avoid crushing the mids (like `0.45/0.08`).

## Scene tips
- Many depth layers and fast, close foreground parallax.
- Small pure-white or cool highlights against saturated colors and deep darks. Large white areas (sky, cloud sea, mist) force a darker grade.
- Warm yellow highlights may look less punchy in HLG (a taste note, not a failure: warm-highlight clips still lit up).
- 60 fps (ffmpeg `minterpolate`) looked about the same as 24 fps in our tests, even for a fast shot, so 24 fps is fine.

## License
MIT

## Example: cartoon rider (`cartoon_rider/`)
A procedural cartoon remake of a "rider going down a sunny grass slope toward a cliff city" shot, used to test the HDR look on non-AI footage. Needs only numpy and Pillow.
```bash
python3 cartoon_rider/render.py sdr.mp4            # 1080p60, 7 s SDR master (~4 min on 4 cores)
python3 cartoon_rider/render.py prev --preview 0 3 # PNG stills only
CURVE="0/0 0.25/0.10 0.5/0.26 0.75/0.52 0.9/0.80 1/1" bash grade_hlg.sh sdr.mp4 examples/cartoon_rider_hlg.mp4
```
The SDR white point maps to the HLG peak, so the scene keeps pure white for only small areas (cloud tops, the T-shirt, the backlit grass rim) and keeps the sky a deeper blue. Result: median 0.49, 15.7% above 0.75, 8.6% above 0.9. For comparison, the AI reference clip measured median 0.40, 8.5% above 0.75, 2.7% above 0.9.

## Example: bright, pale live-action clip
A 37 s, 854x480 clip with large pale grey backgrounds (SDR median 0.64). The default curve left the median at 0.75 and 51% of pixels above 0.75. This curve pulls the background down so only the moon and the white liquid reach the peak:
```bash
CURVE="0/0 0.1/0.03 0.4/0.09 0.7/0.26 0.9/0.66 1/1" bash grade_hlg.sh in.mp4 out.mp4
```
Result: median 0.59, 15.9% above 0.75, 5.6% above 0.9.

## HDR profile picture (`pfp_hdr.py`)
Makes a 400 × 400, 256-colour Rec.2100 PQ PNG for an X avatar. The PNG carries a cICP chunk (9/16/0/1), a PQ ICC profile built by the script, and one transparent pixel at (0,0). This follows the format xhdr.app reports as keeping HDR on X. Upload it from x.com in a desktop browser, and don't zoom or move it in the crop dialog.
```bash
python3 pfp_hdr.py photo.png xhdr-pfp.png                       # centre square crop
python3 pfp_hdr.py screenshot.png xhdr-pfp.png --circle CX CY R  # cut the circle out of a profile screenshot
```
The grade keeps hue, puts the SDR level at 1.4 × 203 nits, and pushes near-white, low-colour pixels toward `--white` nits. Bright, colourful pixels that aren't skin are lifted up to `1 + --pop` times. Warm hues (skin, lips, hair) don't get the lift. On a cartoon avatar with a pale blue background, this gave a background of about 1,500 nits, a face of 160 nits, and sparkles near 1,900 nits. In a test upload to X, that version kept HDR, but the background clipped to plain white on an iPhone. It needs 8× headroom, more than the X app gave. `--pop 3 --white 1200 --sat 1.6` puts the background near 760 nits (about 4× headroom) and keeps it blue. `--sat` adds colour only where the pop applies.

Pale backgrounds lose their colour when pushed bright, because all three channels hit the display limit together. `--bg 00d47a --bg-nits 800` finds the flat background (the area joined to the image border that is close to the border colour). It repaints that area in a saturated colour, keeps its shading and white sparkles, and sets its brightest channel to `--bg-nits`. A saturated colour stays that colour when it is bright. Green gives the most visible glow, since blue adds little luminance. The script also writes `_16bit.png` (full-precision PQ), `_sdr.png` and `_preview.png` (original next to a nits heat map).

## Example: vertical illustrated animation
A 90 s, 720x1280 animation of white cranes over teal water, golden reeds and an orange cloud (SDR median 0.42). The white cranes take up well under 1% of the frame, so they make good small highlights. This curve deepens the teal water and lifts the top end:
```bash
CURVE="0/0 0.1/0.035 0.3/0.14 0.55/0.38 0.75/0.82 1/1" bash grade_hlg.sh in.mp4 out.mp4
```
Result: 1080x1920, median 0.54, 8.5% above 0.75, 0.6% above 0.9 (the cranes and their reflections).

## Colour-targeted grade (`grade_pop.py`)
A global curve lifts every bright pixel, so a pale sky clips to white before the subject glows. `grade_pop.py` instead gives each pixel one gain, so hue is kept. Pixels inside a hue band that are colourful and not dark get lifted up to `1 + --pop`. Near-white, low-colour pixels rise toward `--white` nits, and everything else stays at `--base` × SDR level. It writes HLG directly (BT.2100 inverse OOTF and OETF, 1000-nit reference), with the same H.264 tags as `grade_hlg.sh`.
```bash
python3 grade_pop.py in.mp4 out.mp4 --hue 20 75 --pop 3      # golden / orange leaves
python3 grade_pop.py in.mp4 out.mp4 --mask mask.mp4          # also write the lift mask for checking
```
Example: a 7 s vertical clip of golden ginkgo trees under a pale cyan sky. The sky (linear 0.82) was brighter than the clouds (0.72). Every curve sent the sky to peak first. With `--hue 20 75 --pop 3`, the yellow canopy and fallen leaves became the highlight (22% above HLG 0.75), and the sky stayed at SDR level.
