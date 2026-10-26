#!/bin/bash
# Decoder-independent truth for the fq benchmark: 128 frames of ffmpeg test patterns, 928x576 yuv444p16le
# (limited range, as newdecode / ld-chroma-decoder output): 32 moving test card, 32 moving zone plate,
# 32 Mandelbrot zoom (fine detail), 32 still PAL 75% bars (a held still); then made PAL-legal (legalise_pal.py).
#   make_patterns.sh <out.yuv>
O=$1; T=$(mktemp -d)
g(){ ffmpeg -v error -y -f lavfi -i "$1" -frames:v 32 -vf "scale=928:576:flags=lanczos,format=yuv444p16le" -color_range tv -f rawvideo "$T/$2.yuv"; }
g "testsrc2=size=928x576:rate=25" a
g "zoneplate=size=928x576:rate=25:ku=0:kv=0:kx2=12:ky2=12:kt=6" b
g "mandelbrot=size=928x576:rate=25:start_scale=0.5:end_scale=0.3" c
g "pal75bars=size=928x576:rate=25" d
cat "$T"/a.yuv "$T"/b.yuv "$T"/c.yuv "$T"/d.yuv > "$T/raw.yuv"
. "$(dirname "$0")/../config.sh"
"$PY" "$(dirname "$0")/legalise_pal.py" "$T/raw.yuv" "$O"; rm -rf "$T"
ls -la "$O"
