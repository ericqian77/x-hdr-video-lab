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
Needs ffmpeg built with zscale (libzimg), and Python with numpy.

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
