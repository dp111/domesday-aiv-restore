#!/bin/bash
# Chunked newdecode (2026-10-18, generalised 2026-10-19 from nd_chunked_north.sh).
#   nd_chunked.sh <side> <scratch dir with doc.tbc> <audio pcm> [tag v6]  -> $V7_OUT/<side>.<tag>.mkv
# Origin: The v5 chain decoded the side in one 5.5 h run and threw
# the partial output away on a failure; North died twice on GPU errors (19:31 CUDA OOM at frame
# 29824, 19:54 cuFFT internal error at frame 832). User: "why can't you continue from just before
# the crash?"
# Each chunk of 5120 frames (a multiple of newdecode's 64-frame window) is decoded with 64 frames of
# lead-in/out, trimmed to its own range and kept as ndc/chunk_NN.mkv; a finished chunk is reused, so
# a crash costs one chunk. Test 2026-10-18 (frames 896-1216 vs 960-1216): only the first 8 frames
# after a run start differ materially; elsewhere run-to-run GPU noise (max ~1 8-bit level).
# newdecode runs under nd_gpuwrap.py (frees the PyTorch GPU cache before cuFFT flow; see there).
# Then: concat + audio -> $V7_OUT/<side>.<tag>.mkv, a decode log with a summed done line and satratio.
# The chunks are tied to the doc.tbc they came from ($C/doc.id: its size + mtime); a new doc.tbc (restack)
# discards them, and chunks without that stamp are kept only if newer than doc.tbc.
# GPU exclusivity is atomic: the chunk loop holds an exclusive flock on $ND_GPULOCK ($WORK/state/gpu.lock),
# inherited by newdecode, so two sides cannot both start a newdecode inside gpugate.sh's polling window;
# it is released before the concat, so the next side's decode overlaps that.
set -u
set -o pipefail
S=$1 SCR=$2 APCM=$3 TAG=${4:-v6}
B=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$B/../config.sh"
CD=${ND_CD:-$DAR/newdecode}   # ND_EXTRA="--dropout_mask 1": newdecode/README_dropout.txt
OUT=$V7_OUT
C=$SCR/ndc
LOG=${ND_LOG:-$WORK/pipeline.log}
MKV=$OUT/$S.$TAG.mkv DLOG=$OUT/${S}_${TAG}_decode.log
# ND_MASTER=p10 (2026-10-24, user: "928, 10-bit 4:2:2"): store the master as yuv422p10le, co-sited (left) chroma,
# Lanczos chroma filter with accurate rounding (round trip vs 16-bit 4:4:4: Y 71 dB = 10-bit rounding; chroma
# error 0.25% of signal below 1.5 MHz, 92% of it above 4.2 MHz where PAL has no chroma), ~23 GB/side vs ~92.
# Read back with newdecode/stillweave.py's matching Lanczos upsample.  Default: yuv444p16 as before.
MVF= MTAGS=
if [ "${ND_MASTER:-}" = p10 ]; then
  MVF=",scale=flags=lanczos+accurate_rnd+full_chroma_int:in_range=tv:out_range=tv:out_chroma_loc=left,format=yuv422p10le"
  MTAGS="-color_range tv -colorspace bt470bg -color_primaries bt470bg -color_trc bt709 -chroma_sample_location left"
fi
CH=${ND_CH:-5120} PAD=64 TRIES=3   # 10-21: 2560 chunks peaked ~32 GB too, so halving buys nothing
export ND_FORCE_T3=1   # chunk 02/05 each had a lone interlaced frame left unsolved without it (see nd_gpuwrap.py)
log(){ echo "$(date "+%m-%d %H:%M:%S") ${TAG^^} $*" >> "$LOG"; }
die(){ log "STOPPING (chunked decode): $*"; exit 1; }
nframes(){ ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$1" 2>/dev/null; }
mkdir -p "$C"
N=$(( $(stat -c %s "$SCR/doc.tbc") / (1135 * 313 * 2 * 2) ))
NC=$(( (N + CH - 1) / CH ))
DID=$(stat -c '%s %Y' "$SCR/doc.tbc")
if [ -s "$C/doc.id" ] && [ "$(cat "$C/doc.id")" != "$DID" ]; then
  log "chunked newdecode $S: doc.tbc changed since the chunks were made - discarding them"
  rm -rf "$C"/chunk_* "$C"/sat_*.txt "$C"/cls_*.txt "$C"/w[0-9]*
elif [ ! -s "$C/doc.id" ]; then
  for m in "$C"/chunk_*.mkv; do
    [ -e "$m" ] && [ "$m" -ot "$SCR/doc.tbc" ] && { K=${m##*/chunk_}; K=${K%.mkv}; rm -f "$m" "$C/chunk_$K.log" "$C/sat_$K.txt" "$C/cls_$K.txt"; log "chunk $K older than doc.tbc - discarded"; }
  done
fi
echo "$DID" > "$C/doc.id"
log "chunked newdecode $S: $N frames in $NC chunks of $CH (+-$PAD lead) -> $C (decoder $CD ${ND_EXTRA:-})"
GL=${ND_GPULOCK:-$WORK/state/gpu.lock}
mkdir -p "$(dirname "$GL")"; exec 9> "$GL" || die "cannot open GPU lock $GL"
if ! flock -n 9; then
  log "chunked newdecode $S: waiting for the GPU (another newdecode holds $GL)"
  flock 9 || die "GPU lock $GL failed"
  log "chunked newdecode $S: GPU lock taken"
fi

for ((k = 0; k < NC; k++)); do
  s=$((k * CH)); e=$((s + CH)); [ $e -gt $N ] && e=$N
  ds=$((s - PAD)); [ $ds -lt 0 ] && ds=0
  de=$((e + PAD)); [ $de -gt $N ] && de=$N
  K=$(printf %02d $k); M=$C/chunk_$K.mkv L=$C/chunk_$K.log
  if [ "$(nframes "$M")" = $((e - s)) ] && grep -q '\[newdecode\] done' "$L" 2>/dev/null; then continue; fi
  for ((t = 1; t <= TRIES; t++)); do
    rm -rf "$C/w$K"; rm -f "$M"
    ZIP_DUMP=$C/w$K/classes.txt "$MLPY" "$B/nd_gpuwrap.py" "$CD/newdecode.py" "$SCR/doc.tbc" - --scratch "$C/w$K/nd" --video 1 ${ND_EXTRA:-} --start $ds --frames $((de - ds)) 2> "$L" \
      | ffmpeg -hide_banner -loglevel error -y -i - -vf "select='between(n,$((s - ds)),$((e - ds - 1)))',setpts=N/25/TB$MVF" -fps_mode passthrough \
          $MTAGS -c:v ffv1 -level 3 -slices 16 -slicecrc 1 "$M" \
      && [ "$(nframes "$M")" = $((e - s)) ] && grep -q '\[newdecode\] done' "$L" && break
    log "chunk $K ($s-$e) try $t failed: $(grep -v -i warn "$L" | tail -1 | cut -c1-200)"
    [ $t = $TRIES ] && die "chunk $K failed $TRIES times (see $L)"
    sleep 60
  done
  awk -v s=$s -v e=$e '$1 >= s && $1 < e' "$C/w$K/nd/satratio.txt" > "$C/sat_$K.txt" 2>/dev/null || : > "$C/sat_$K.txt"
  # per-frame class log (newdecode ZIP_DUMP) for the quality record (quality/quality_db.py nd)
  awk -v s=$s -v e=$e '$1 >= s && $1 < e' "$C/w$K/classes.txt" > "$C/cls_$K.txt" 2>/dev/null || : > "$C/cls_$K.txt"
  rm -rf "$C/w$K"
  log "chunk $K ($s-$e) OK $(stat -c%s "$M") bytes"
done
flock -u 9; exec 9>&-                  # GPU free: the concat + checks below are CPU / disk only

ls "$C"/chunk_*.mkv | sed "s/.*/file '&'/" > "$C/list.txt"
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$C/list.txt" -f s16le -ar 44100 -ac 2 -i "$APCM" \
    -map 0:v -map 1:a -c:v copy -c:a flac "$MKV.tmp.mkv" || die "concat failed"
[ "$(nframes "$MKV.tmp.mkv")" = $N ] || die "concat has $(nframes "$MKV.tmp.mkv") frames, want $N"
mv -f "$MKV.tmp.mkv" "$MKV"
cat "$C"/sat_*.txt > "$OUT/${S}_${TAG}_satratio.txt"
cat "$C"/cls_*.txt > "$OUT/${S}_${TAG}_classes.txt" 2>/dev/null || :
# decode log for the chain: the chunk logs + one done line with stats summed over chunks (incl. lead frames)
{ cat "$C"/chunk_*.log
  "$MLPY" - "$C" <<'E'
import ast, glob, re, sys
tot = {}
for f in sorted(glob.glob(sys.argv[1] + '/chunk_*.log')):
    d = ast.literal_eval(re.search(r"frames (\{.*\})", [l for l in open(f) if '[newdecode] done' in l][-1])[1])
    for k, v in d.items(): tot[k] = tot.get(k, 0) + v
print('[newdecode] done: chunked (%d chunks); frames %s' % (len(glob.glob(sys.argv[1] + '/chunk_*.log')), tot))
E
} > "$DLOG" 2>&1
done_l=$(grep '\[newdecode\] done: chunked' "$DLOG")
ni=$(echo "$done_l" | grep -o "'interlaced': [0-9]*" | grep -o '[0-9]*$')
nv=$(echo "$done_l" | grep -o "'video_solved': [0-9]*" | grep -o '[0-9]*$')
[ -n "$ni" ] && [ "$ni" = "${nv:-0}" ] || die "inline video solve incomplete (interlaced $ni, solved $nv)"
log "newdecode OK (chunked): $done_l; $(basename "$MKV") $(stat -c%s "$MKV") bytes"
