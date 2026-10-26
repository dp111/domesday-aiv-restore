#!/bin/bash
# Per-frame loss of the final PVF encode against its own source.
#   pvf_quality.sh <video.pvf> <master.mkv> <out prefix> [pvf build log]  -> <prefix>.psnr.log, <prefix>.ssim.log
# Reference = exactly what make_pvfv2 encoded: stillweave.py of the master, the build's padding crop (read
# from its log: "cropping 3 padding samples each side to 922"), scale=832:576 lanczos, yuv420p.
# Both sides get setpts=N so frames pair by index (ffmpeg pairs by timestamp, and a raw .h264 has none).
set -eu
PVF=$1 MKV=$2 P=$3
. "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
CROP=
if [ -n "${4:-}" ]; then
  c=$(grep -o 'cropping [0-9]* padding samples each side to [0-9]*' "$4" | tail -1)
  [ -n "$c" ] && CROP="crop=$(echo "$c" | awk '{print $8}'):in_h:$(echo "$c" | awk '{print $2}'):0,"
fi
ffmpeg -hide_banner -loglevel error -y \
  -framerate 25 -f h264 -i <("$PY" "$DAR/quality/pvf_frames.py" "$PVF") \
  -i <("$PY" "$DAR/newdecode/stillweave.py" "$MKV" 2> /dev/null) \
  -lavfi "[0:v]setpts=N/25/TB,format=yuv420p,split[a1][a2];[1:v]${CROP}scale=832:576:flags=lanczos,format=yuv420p,setpts=N/25/TB,split[b1][b2];[a1][b1]psnr=stats_file=$P.psnr.log;[a2][b2]ssim=stats_file=$P.ssim.log" \
  -f null -
