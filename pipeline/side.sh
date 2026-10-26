#!/bin/bash
# One side, start to finish:  side.sh <side>   (side names: sides.conf; run.sh starts every side at once)
#   wait for the side's capture decodes (decodeq.sh)
#   -> stack: ld-decode lddecode.stack --cross_master [--clv] (2 stack slots)
#   -> automatic stack QC -> squelch audio -> ld-dropout-correct
#   -> chunked newdecode (newdecode/, --dropout_mask 1; 1 GPU slot) -> master <side>.v8.mkv, yuv422p10le
#   -> PVF (still-weave, CRF 17 [18 if >= 4 GB], dedupe) -> BeebVFS slot, previous PVF kept as video.pvf.pre-v8
#   -> per-frame quality record in quality/quality.db
# Slots are handed out in side order (an earlier side waiting for a slot goes first).
# The stack is gated automatically on frame count and, if V7_OLDF holds an earlier stack of the side, on
# stack_quality.py's predicted noise; a failed gate HOLDs only this side ($V7/HOLD.<side>, remove it to continue).
# $V7/AUTODELETE: delete this side's capture decodes once its stack + dropout-correct are done.
# Resumable: the stack, doc.tbc, chunks and MKV are reused.  Log prefix "AIV <side>".
set -u
set -o pipefail
. "$(dirname "${BASH_SOURCE[0]}")/conf.sh"
SIDE=$1
ORD=$(printf %02d "$(side_field "$SIDE" 1)"); [ "$ORD" != 00 ] || { echo "unknown side $SIDE"; exit 2; }
SLOTS=$(side_field "$SIDE" 3) CLV=$(side_field "$SIDE" 4)
SLOT=$V7_SLOTROOT/BeebVFS${SLOTS%%,*}
OUT=$V7_OUT OLDF=$V7_OLDF
SCR=$V7_SCRROOT/$SIDE
STK=$LD/lddecode/stack.py ND=$DAR/newdecode PVT=$DAR/pvf
HOLDF=$V7/HOLD.$SIDE SLOTLOCK=
log(){ v7log "$SIDE $*"; }
release(){ [ -n "$SLOTLOCK" ] && rm -rf "$SLOTLOCK"; SLOTLOCK=; }
trap release EXIT
die(){ log "STOPPING: $*"; exit 1; }
sethold(){ touch "$HOLDF"; log "HOLD set: $*"; while [ -e "$HOLDF" ]; do sleep 60; done; log "HOLD released"; }
nframes(){ ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$1" 2>/dev/null; }
acquire(){ # <resource> <slots>: wait for a free slot; earlier sides first
  local w=$V7/wait/$1.$ORD i first
  mkdir -p "$V7/wait" "$V7/locks"; touch "$w"
  while :; do
    first=$(ls "$V7/wait" | grep "^$1\." | sort | head -1)
    if [ "$first" = "$1.$ORD" ] && [ ! -e "$V7/PAUSE_$1" ]; then
      for ((i = 1; i <= $2; i++)); do
        # a second GPU decoder (+1.5x throughput, ~15 GB RAM each) only with room to spare
        [ "$1" = gpu ] && [ $i -gt 1 ] && [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt 20 ] && continue
        if mkdir "$V7/locks/$1.$i" 2>/dev/null; then
          SLOTLOCK=$V7/locks/$1.$i; echo "$SIDE $$" > "$SLOTLOCK/owner"; rm -f "$w"; return 0
        fi
      done
    fi
    sleep 60
  done
}
mkdir -p "$SCR"
log "=== v8 start: slot $(basename "$SLOT"), $([ "$CLV" = 1 ] && echo CLV || echo CAV), $(side_ldfs "$SIDE" | wc -l) captures ==="

# ---- captures --------------------------------------------------------------------------
ins=() bad=()
for ldf in $(side_ldfs "$SIDE"); do
  ca=$CAPS/$SIDE/$(cap_of "$ldf")
  until [ -e "$ca.done" ] || [ -e "$ca.failed" ]; do sleep 120; done
  if [ -e "$ca.done" ]; then ins+=("$(cat "$ca.path").tbc"); else bad+=("$(cap_of "$ldf")"); fi
done
[ ${#bad[@]} = 0 ] || log "captures that failed to decode, left out: ${bad[*]}"
[ ${#ins[@]} -gt 0 ] || die "no capture decoded"
log "${#ins[@]} captures decoded"

# ---- stack -----------------------------------------------------------------------------
STKO=$OUT/$SIDE.xm4 SL=$OUT/${SIDE}_stack_xm4.log
if [ ${#ins[@]} = 1 ]; then
  STKO=${ins[0]%.tbc}; log "single capture: no stack, using $STKO.tbc directly"
elif grep -q "^\[stack\] wrote" "$SL" 2>/dev/null && [ -s "$STKO.tbc" ]; then
  log "xm4 stack already complete, reusing"
else
  for try in 1 2; do
    acquire stack 2
    log "xm4 stack start (try $try; ${#ins[@]} captures$([ "$CLV" = 1 ] && echo ', --clv'), stacker $(md5sum "$STK" | cut -c1-8))"
    rm -rf "$SCR/repair_dump"; mkdir -p "$SCR/repair_dump"
    ( cd "$LD" && LDSTACK_FRAME_STATS="$OUT/${SIDE}_stack_frames.tsv" LDSTACK_REPAIR_DUMP="$SCR/repair_dump" "$PY" -m lddecode.stack --PAL --jobs 16 --cross_master \
        $([ "$CLV" = 1 ] && echo --clv) -o "$STKO" "${ins[@]}" > "$SL" 2>&1 )
    release
    grep -q "^\[stack\] wrote" "$SL" && [ -s "$STKO.tbc" ] && break
    log "xm4 stack try $try failed: $(tail -n 2 "$SL" | tr '\n' ' ' | cut -c1-300)"
    [ $try = 2 ] && die "xm4 stack failed twice (see $SL)"
    sleep 120
  done
  log "xm4 stack done: $(grep '^\[stack\] wrote' "$SL" | grep -o '([0-9]* frames.*')"
  log "masters: $(grep 'master=' "$SL" | tr -s ' ' | tr '\n' ';' | cut -c1-600)"
fi
[ -s "$STKO.tbc.json" ] || "$PY" "$LD/scripts/db2json.py" "$STKO.tbc.db" > /dev/null 2>&1
[ -s "$STKO.tbc.json" ] || die "no stack json"
NFR=$(( $(stat -c %s "$STKO.tbc") / FB ))

# ---- automatic stack QC (frame count and predicted noise against an earlier stack, if any) ----
if [ ! -e "$SCR/stackqc.ok" ]; then
  qc=ok
  ON=$(( $(stat -c %s "$OLDF/$SIDE.tbc" 2>/dev/null || echo 0) / FB ))
  [ "$ON" -gt 0 ] && [ "$NFR" -lt $((ON - 100)) ] && qc="only $NFR frames against the earlier stack's $ON"
  OL=$OLDF/${SIDE}_stack_redo.log; [ -s "$OL" ] || OL=$OLDF/${SIDE}_stack.log
  if [ ${#ins[@]} -gt 1 ] && [ -s "$OL" ]; then
    q=$("$PY" "$B/stack_quality.py" "$OL" "$SL" 2>&1 | tr '\n' ' ')
    log "stack quality (predicted noise; earlier then new): $q"
    B=$B "$PY" - "$OL" "$SL" <<'E' || qc="predicted noise worse than the earlier stack ($q)"
import os, sys; sys.path.insert(0, os.environ['B'])
from stack_quality import predicted
o, n = predicted(sys.argv[1])[0], predicted(sys.argv[2])[0]
sys.exit(0 if n <= o * 1.05 else 1)
E
  fi
  log "stack QC: $NFR frames (earlier $ON): $qc"
  [ "$qc" = ok ] || sethold "stack QC: $qc ($STKO.tbc)"
  touch "$SCR/stackqc.ok"
fi

# ---- audio, dropout-correct --------------------------------------------------------------
APCM=$OUT/${SIDE}_xm4_squelched.pcm
[ -s "$APCM" ] || "$PY" "$LD/scripts/squelch_pcm.py" "$STKO.pcm" "$APCM" > "$OUT/${SIDE}_xm4_squelch.log" 2>&1
[ -s "$APCM" ] || APCM=$STKO.pcm
if [ "$(stat -c %s "$SCR/doc.tbc" 2>/dev/null)" = "$(stat -c %s "$STKO.tbc")" ] && grep -q "Total concealments" "$SCR/doc.log" 2>/dev/null; then
  log "dropout-corrected stack already in scratch, reusing"
elif [ "$(nframes "$OUT/$SIDE.v8.mkv")" != "$NFR" ]; then
  log "dropout-correct"
  ld-dropout-correct --input-json "$STKO.tbc.json" --output-json "$SCR/doc.tbc.json" \
      "$STKO.tbc" "$SCR/doc.tbc" > "$SCR/doc.log" 2>&1 || die "dropout-correct failed"
  log "$(grep -i 'Total concealments' "$SCR/doc.log" | tail -1)"
fi
if [ -e "$V7/AUTODELETE" ] && [ ${#ins[@]} -gt 1 ]; then
  for t in "${ins[@]}"; do rm -f "$t" "$t.db" "$t.json" "${t%.tbc}.pcm" "${t%.tbc}.efm"; done
  log "capture decodes deleted (stack + doc done)"
fi

# ---- model-based decode (GPU) ----------------------------------------------------------------
MKV=$OUT/$SIDE.v8.mkv
if [ "$(nframes "$MKV")" = "$NFR" ] && grep -q '\[newdecode\] done' "$OUT/${SIDE}_v8_decode.log" 2>/dev/null; then
  log "$SIDE.v8.mkv already complete, reusing"
else
  acquire gpu "${V7_GPUS:-2}"
  log "chunked newdecode (--dropout_mask 1, master yuv422p10le) -> $MKV"
  ND_MASTER=p10 ND_CD=$ND ND_EXTRA="--dropout_mask 1" "$B/nd_chunked.sh" "$SIDE" "$SCR" "$APCM" v8 \
    || die "chunked newdecode failed"
  release
fi
"$PY" - "$OUT/${SIDE}_v8_satratio.txt" <<'E' | while read -r l; do log "decode QC: $l"; done
import sys, numpy as np
try:
    r = np.loadtxt(sys.argv[1], ndmin=2)
    print('video-solved %d, satratio p5 %.3f median %.3f, fallbacks %d' % (len(r), np.percentile(r[:, 1], 5), np.median(r[:, 1]), r[:, 2].sum()))
except Exception as e:
    print('no satratio (%s)' % e)
E
log "decode: $(grep '\[newdecode\] done: chunked' "$OUT/${SIDE}_v8_decode.log" | cut -c1-300)"

# ---- PVF --------------------------------------------------------------------------------------
F=${TMPDIR:-/tmp}/${SIDE}_v8_weave.y4m PL=$OUT/${SIDE}_v8_pvf.log d=$SLOT crf=17
build_pvf(){ # <crf>
  rm -f "$F" "$d/.video.pvf.tmp"; mkfifo "$F" || die "cannot create FIFO $F"
  "$PY" "$ND/stillweave.py" "$MKV" --log "$OUT/${SIDE}_v8_weave.log" > "$F" 2>> "$PL" &
  local wp=$!
  echo "=== make_pvfv2 --crf $1 ===" >> "$PL"
  nice -n 10 "$PY" "$PVT/make_pvfv2.py" "$F" "$d/.video.pvf.tmp" --dedupe --crf "$1" --probe-from "$MKV" \
      --tbc-json "$STKO.tbc.json" --audio-input "$APCM" --audio-input-rate 44100 \
      --vf "scale=832:576:flags=lanczos" >> "$PL" 2>&1 \
   && nice -n 10 "$PY" "$PVT/verify_pvf.py" "$d/.video.pvf.tmp" >> "$PL" 2>&1
  local rc=$?; [ $rc = 0 ] || kill $wp 2>/dev/null
  wait $wp; rm -f "$F"; return $rc
}
if grep -q "PVF v8 OK" <(grep "AIV $SIDE " "$LOG"); then
  log "PVF already installed"
else
  acquire pvf 1; mkdir -p "$d"
  log "PVF v8 (still-weave, CRF 17, dedupe) -> $d"
  if build_pvf 17 && [ "$(stat -c%s "$d/.video.pvf.tmp")" -lt 4000000000 ]; then :
  else
    log "CRF 17 PVF failed or >= 4 GB ($(stat -c%s "$d/.video.pvf.tmp" 2>/dev/null) bytes) - rebuilding at CRF 18"
    crf=18; build_pvf 18 || { rm -f "$d/.video.pvf.tmp"; die "PVF v8 failed (see $PL)"; }
  fi
  release
  OLDPVF=video.pvf.pre-v8
  if [ -e "$d/video.pvf" ]; then
    [ -e "$d/$OLDPVF" ] && die "$d/$OLDPVF already exists - not overwriting (new PVF left at .video.pvf.tmp)"
    mv -f "$d/video.pvf" "$d/$OLDPVF"
  fi
  mv -f "$d/.video.pvf.tmp" "$d/video.pvf"
  log "PVF v8 OK (CRF $crf) $(stat -c%s "$d/video.pvf") bytes; $(grep '^dedupe: .*share' "$PL" | tail -1); previous kept as $OLDPVF"
  for n in $(echo "$SLOTS" | tr ',' ' ' | cut -d' ' -f2-); do          # further slots get a copy
    [ "$n" = "${SLOTS%%,*}" ] && continue
    e=$V7_SLOTROOT/BeebVFS$n; mkdir -p "$e"
    cp "$d/video.pvf" "$e/.video.pvf.tmp" && cmp -s "$d/video.pvf" "$e/.video.pvf.tmp" || die "copy to $e failed"
    o=video.pvf.pre-v8
    if [ -e "$e/video.pvf" ]; then [ -e "$e/$o" ] && die "$e/$o exists"; mv -f "$e/video.pvf" "$e/$o"; fi
    mv -f "$e/.video.pvf.tmp" "$e/video.pvf"; log "PVF copied to $(basename "$e") (previous kept as $o)"
  done
fi
rm -rf "$SCR/doc.tbc" "$SCR/doc.tbc.json" "$SCR/ndc"

# ---- per-frame quality record (quality/quality_db.py).  Holds no slot; read-only on the stack, master and PVF.
FQ=$DAR/quality QL=$OUT/${SIDE}_v8_quality.log
if ! grep -q "quality record done" <(grep "AIV $SIDE " "$LOG"); then
  log "quality record: stack frames, blank-line noise + VITS, newdecode, PVF encode loss"
  {
    RUN=$("$PY" $FQ/quality_db.py run "$SIDE" v8 "ld=$(git -C "$LD" log --oneline -1 | cut -c1-8)" \
          "ld_opts=$LD_OPTS" "stacker=$(md5sum "$STK" | cut -c1-8)" "nd=$(md5sum "$ND/newdecode.py" | cut -c1-8)" \
          master=yuv422p10le "captures=${#ins[@]}" "pvf_crf=$crf")
    [ -s "$OUT/${SIDE}_stack_frames.tsv" ] && "$PY" $FQ/quality_db.py stack "$RUN" "$OUT/${SIDE}_stack_frames.tsv"
    nice -n 10 "$PY" $FQ/fq_side.py "$STKO.tbc" "$OUT/${SIDE}_v8_fields.csv" --vits \
      && "$PY" $FQ/quality_db.py fields "$RUN" "$OUT/${SIDE}_v8_fields.csv"
    "$PY" $FQ/quality_db.py nd "$RUN" "$OUT/${SIDE}_v8_satratio.txt" "$OUT/${SIDE}_v8_classes.txt"
    nice -n 10 $FQ/pvf_quality.sh "$SLOT/video.pvf" "$MKV" "$OUT/${SIDE}_v8_pvfq" "$PL" \
      && "$PY" $FQ/quality_db.py pvf "$RUN" "$OUT/${SIDE}_v8_pvfq"
    "$PY" $FQ/quality_db.py summary "$RUN"
  } > "$QL" 2>&1
  log "quality record done: $(tail -1 "$QL" | cut -c1-400)"
fi
log "=== v8 done: $MKV + $(basename "$SLOT")/video.pvf ==="
