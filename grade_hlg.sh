#!/bin/bash
set -euo pipefail
SRC=${1:?src}; OUT=${2:?out}
CURVE=${CURVE:-"0/0 0.45/0.31 0.75/0.71 1/1"}
PRE=${PRE:-""}
VF="scale=1920:1080:flags=lanczos,zscale=tin=bt709:min=bt709:pin=bt709:t=linear:npl=1000,format=gbrpf32le,${PRE}curves=all='${CURVE}',zscale=pin=bt709:tin=linear:p=bt2020:t=arib-std-b67:m=bt2020nc:r=tv:npl=1000,format=yuv420p"
ffmpeg -y -i "$SRC" -vf "$VF" -map 0:v:0 -map 0:a? \
  -c:v libx264 -profile:v high -pix_fmt yuv420p -b:v 10M -maxrate 12M -bufsize 20M \
  -color_primaries bt2020 -color_trc arib-std-b67 -colorspace bt2020nc -color_range tv \
  -x264-params "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc" \
  -c:a aac -b:a 128k -movflags +faststart "$OUT"
