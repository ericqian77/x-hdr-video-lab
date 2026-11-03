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
`SIZE` sets the output size (default `1920:1080`); set it to match portrait clips, e.g. `SIZE=1080:1620`.

### From a still image
`still_to_motion.py` turns a still (poster, AI image) into an 8 s seamless loop made for HDR pop: a slow orbit camera (push-in, roll, float), pulsing bloom with white-hot cores on lamps and windows, a gloss sweep over lettering and bright rims (skin is left alone), and small cool-white sparkles.
```bash
python3 still_to_motion.py poster.jpg sdr.mp4            # 1080 wide, height follows the image
SIZE=1080:1620 PRE="eq=brightness=0.03," CURVE="0/0 0.32/0.28 0.55/0.60 0.78/0.82 1/1" bash grade_hlg.sh sdr.mp4 hdr.mp4
```
To make the subject float off the background, give it a matte. The subject then moves on its own layer (the hole behind it is filled), gets a pulsing rim light, and snow falls both behind and in front of it:
```bash
pip install "rembg[cpu]" && rembg i -om poster.png mask.png      # white = subject
python3 still_to_motion.py poster.png sdr.mp4 --mask mask.png --snow 500 --rays 0.6,0
SIZE=1080:1620 PRE="eq=brightness=0.03," CURVE="0/0 0.25/0.24 0.5/0.57 0.7/0.81 0.85/0.95 1/1" bash grade_hlg.sh sdr.mp4 hdr.mp4
```
The last curve keeps a night scene's darks and pushes the top end harder; on a dark snowy poster (SDR median 0.25) it gave median 0.50, 15% above 0.75.

Options: `--seconds`, `--fps`, `--bloom`, `--sweep`, `--sweep-box x0,y0,x1,y1`, `--sparkles N`, `--snow N`, `--rays x,y`, `--emissive`, `--mask`, `--float`, `--rim`, `--bg-dim`, `--frames 0,60` (save PNG previews instead of encoding).
On a 1024x1536 cafe poster this graded to median 0.56, 27% above 0.75 with the brighter preset (0.45 / 21% with the default curve).

Needs ffmpeg built with zscale (libzimg), and Python with numpy.

A result that looked good: median luma about 0.54, about 24% of pixels above 0.75. A median under 0.4 looked too dark. Avoid crushing the mids (like `0.45/0.08`).

## Scene tips
- Many depth layers and fast, close foreground parallax.
- Small pure-white or cool highlights against saturated colors and deep darks. Large white areas (sky, cloud sea, mist) force a darker grade.
- Warm yellow highlights may look less punchy in HLG (a taste note, not a failure: warm-highlight clips still lit up).
- 60 fps (ffmpeg `minterpolate`) looked about the same as 24 fps in our tests, even for a fast shot, so 24 fps is fine.

## License
MIT
