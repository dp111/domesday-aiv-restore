#!/bin/bash
# Per-frame loss of the final PVF encode against its own source.
#   pvf_quality.sh <video.pvf> <master.mkv> <out prefix> [pvf build log]  -> <prefix>.psnr.log, <prefix>.ssim.log
# Reference = exactly what make_pvfv2 encoded: stillweave.py of the master, the build's padding crop (read
# from its log: "cropping 3 padding samples each side to 922"), scale=832:576 lanczos, yuv420p.
# Fails loudly unless the PVF's frames and the master's pair up 1:1: the PVF header's frame
# count must equal the master's, both are decoded to raw yuv420p (no frame-rate conversion) and compared
# frame n with frame n by pvf_rawcmp.py, which fails if either stream is short or either decode fails.
# (ffmpeg's psnr / ssim filters pair by timestamp and repeat the last frame of a short input, so a
# missing frame went unseen.)
set -eu
set -o pipefail
PVF=$1 MKV=$2 P=$3
. "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
FQ=$DAR/quality SW=$DAR/newdecode/stillweave.py
CROP=
if [ -n "${4:-}" ]; then
  c=$(grep -o 'cropping [0-9]* padding samples each side to [0-9]*' "$4" | tail -1)
  [ -n "$c" ] && CROP="crop=$(echo "$c" | awk '{print $8}'):in_h:$(echo "$c" | awk '{print $2}'):0,"
fi
read -r FC W H < <("$PY" -c 'import struct, sys; h = struct.unpack("<16I", open(sys.argv[1], "rb").read(64)); print(h[6], h[2], h[3])' "$PVF")
MF=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$MKV")
[ "$FC" = "$MF" ] || { echo "pvf_quality: FAIL: PVF has $FC frames, master $MF - they do not pair up" >&2; exit 1; }
q(){ printf '%q' "$1"; }
TEST="$(q "$PY") $(q "$FQ/pvf_frames.py") $(q "$PVF") | ffmpeg -hide_banner -loglevel error -f h264 -i - -fps_mode passthrough -f rawvideo -pix_fmt yuv420p -"
REF="$(q "$PY") $(q "$SW") $(q "$MKV") --expect-frames $MF | ffmpeg -hide_banner -loglevel error -i - -vf $(q "${CROP}scale=$W:$H:flags=lanczos,format=yuv420p") -fps_mode passthrough -f rawvideo -pix_fmt yuv420p -"
"$PY" "$FQ/pvf_rawcmp.py" "$W" "$H" "$P" "$TEST" "$REF" --expect "$FC" \
  || { echo "pvf_quality: FAIL: PVF / master frame comparison (see above)" >&2; exit 1; }
