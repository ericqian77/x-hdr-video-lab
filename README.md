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

A result that looked good: median luma about 0.54, about 24% of pixels above 0.75. A median under 0.4 looked too dark. Avoid crushing the mids (like `0.45/0.08`).

## Stills to motion (`animate_stills.py`)
Turns still images into a photographic-looking clip without a generative video model: every output pixel is resampled from the source still. Motion comes from a slow camera dolly with depth parallax (Depth Anything V2 Small, ONNX), travelling water ripples with caustic focusing derived from the wave height field, and subtle per-scene subject motion (spiral rotation, sphere spin with pulsing inner glow, flower sway) that is mirrored into the reflection. Scene settings (horizon row, crop path, masks) are in `SCENES` at the top of the script and are tuned for `examples/stills/`.
```bash
pip install onnxruntime opencv-python-headless numpy
curl -L -o da2s.onnx https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx
python3 animate_stills.py --images examples/stills --depth-model da2s.onnx --out master_sdr.mkv
CURVE="0/0 0.30/0.25 0.55/0.58 0.75/0.84 1/1" bash grade_hlg.sh master_sdr.mkv out_hlg.mp4
# preview single frames as PNG instead of rendering the clip
python3 animate_stills.py --images examples/stills --depth-model da2s.onnx --out frame --frames 0,100,300
```
For the example stills (19 s, 24 fps, about 7 min on 4 CPU cores) that curve gave median luma 0.55 with 17% of pixels above 0.75. Deep blue sky keeps median luma lower than in brighter scenes.

## Scene tips
- Many depth layers and fast, close foreground parallax.
- Small pure-white or cool highlights against saturated colors and deep darks. Large white areas (sky, cloud sea, mist) force a darker grade.
- Warm yellow highlights may look less punchy in HLG (a taste note, not a failure: warm-highlight clips still lit up).
- 60 fps (ffmpeg `minterpolate`) looked about the same as 24 fps in our tests, even for a fast shot, so 24 fps is fine.

## License
MIT
