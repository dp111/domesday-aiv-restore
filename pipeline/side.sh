#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# One side, start to finish:  side.sh <side>   (side names: sides.conf; run.sh starts every side at once)
#   wait for the side's capture decodes (decodeq.sh)
#   -> stack: ld-decode lddecode.stack --cross_master [--clv] (2 stack slots)
#   -> automatic stack QC -> squelch + mono-combine audio -> ld-dropout-correct
#   -> chunked newdecode (newdecode/, --dropout_mask 1; 1 GPU slot) -> master <side>.v8.mkv, yuv422p10le
#   -> PVF (still-weave, CRF 17 [18, 19, 20 while >= 4 GB], dedupe) -> BeebVFS slot, previous PVF kept as video.pvf.pre-v8
#   -> per-frame quality record in quality/quality.db
# Slots are handed out in side order (an earlier side waiting for a slot goes first).
# The stack is gated automatically on frame count and, if V7_OLDF holds an earlier stack of the side, on
# stack_quality.py's predicted noise; a failed gate HOLDs only this side ($V7/HOLD.<side>, remove it to continue).
# $V7/AUTODELETE: delete this side's capture decodes once its stack + dropout-correct are done.
# Resumable: the stack, doc.tbc, chunks and MKV are reused.  Log prefix "AIV <side>".
# Each product made from the stack (stackqc.ok, doc.tbc, MKV, PVF, quality record) carries a stamp of the
# stack it was made from (<product>.stackid: size + mtime of the stack .tbc) and is redone when the stack
# changes; a product without a stamp counts only if newer than the stack.
set -u
set -o pipefail
. "$(dirname "${BASH_SOURCE[0]}")/conf.sh"
SIDE=$1
ORD=$(printf %02d "$(side_field "$SIDE" 1)"); [ "$ORD" != 00 ] || { echo "unknown side $SIDE"; exit 2; }
SLOTS=$(side_field "$SIDE" 3) CLV=$(side_field "$SIDE" 4)
SLOT=$V7_SLOTROOT/BeebVFS${SLOTS%%,*}
OUT=$V7_OUT OLDF=$V7_OLDF
SCR=$V7_SCRROOT/$SIDE
STK=$LD/lddecode/stack.py SW=$DAR/newdecode/stillweave.py PVT=$DAR/pvf
HOLDF=$V7/HOLD.$SIDE SLOTLOCK=
log(){ v7log "$SIDE $*"; }
release(){ [ -n "$SLOTLOCK" ] && rm -rf "$SLOTLOCK"; SLOTLOCK=; }
trap release EXIT
die(){ log "STOPPING: $*"; exit 1; }
sethold(){ touch "$HOLDF"; log "HOLD set: $*"; while [ -e "$HOLDF" ]; do sleep 60; done; log "HOLD released"; }
nframes(){ ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$1" 2>/dev/null; }
side_alive(){ # <pid>: a running side chain (a reused pid is not; unreadable cmdline = EPERM = alive)
  local c; [ -d "/proc/$1" ] || return 1
  c=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null) || return 0
  [[ $c == *side.sh* ]]; }
acquire(){ # <resource> <slots>: wait for a free slot; earlier sides first
  # PAUSE_<res> is a manual pause, PAUSE_<res>.auto the one gpugate.sh sets and clears
  # the wait file holds "<side> <pid>"; one whose pid is gone (side died) is removed instead
  # of blocking every later side.  An EMPTY wait file (no pid) is a manual hold: it is never removed.
  local w=$V7/wait/$1.$ORD i first e p
  mkdir -p "$V7/wait" "$V7/locks"; echo "$SIDE $$" > "$w.tmp$$" && mv -f "$w.tmp$$" "$w"
  while :; do
    first=
    for e in $(ls "$V7/wait" | grep "^$1\.[0-9]*$" | sort); do
      [ "$e" = "$1.$ORD" ] && { first=$e; break; }
      p=$(awk '{print $NF}' "$V7/wait/$e" 2>/dev/null)
      if [ -n "$p" ] && ! side_alive "$p"; then
        rm -f "$V7/wait/$e"; log "stale wait entry $e removed (pid $p gone)"; continue
      fi
      first=$e; break                    # a live earlier side, or a manual hold (empty file)
    done
    if [ "$first" = "$1.$ORD" ] && [ ! -e "$V7/PAUSE_$1" ] && [ ! -e "$V7/PAUSE_$1.auto" ]; then
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
side_caps(){ # capture ids: from the sides.conf capture glob, else the ready-made records $CAPS/<side>/<id>.path
  local x l; l=$(side_ldfs "$SIDE")
  if [ -n "$l" ]; then for x in $l; do cap_of "$x"; done
  else for x in "$CAPS/$SIDE"/*.path; do [ -e "$x" ] && basename "$x" .path; done; fi; }
mkdir -p "$SCR"
log "=== v8 start: slot $(basename "$SLOT"), $([ "$CLV" = 1 ] && echo CLV || echo CAV), $(side_caps | wc -l) captures ==="

# ---- captures --------------------------------------------------------------------------
ins=() bad=()
for c in $(side_caps); do
  ca=$CAPS/$SIDE/$c
  until [ -e "$ca.done" ] || [ -e "$ca.failed" ]; do sleep 120; done
  if [ -e "$ca.done" ]; then ins+=("$(cat "$ca.path").tbc"); else bad+=("$c"); fi
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
SID=$(stat -c '%s %Y' "$STKO.tbc")        # stack identity: a restack changes it
stamped(){ # <product> <stamp>: made from this stack (stamp matches; no / empty stamp: product newer than the stack)
  if [ -s "$2" ]; then [ "$(cat "$2")" = "$SID" ]; else [ -e "$1" ] && [ ! "$1" -ot "$STKO.tbc" ]; fi; }
stamp(){ echo "$SID" > "$1"; }

# ---- automatic stack QC (frame count and predicted noise against an earlier stack, if any) ----
if ! stamped "$SCR/stackqc.ok" "$SCR/stackqc.ok"; then
  [ -e "$SCR/stackqc.ok" ] && log "stack changed since the last stack QC - redoing it and everything made from the stack"
  qc=ok
  ON=$(( $(stat -c %s "$OLDF/$SIDE.tbc" 2>/dev/null || echo 0) / FB ))
  [ "$ON" -gt 0 ] && [ "$NFR" -lt $((ON - 100)) ] && qc="only $NFR frames against the earlier stack's $ON"
  [ "$ON" -gt 0 ] || log "stack QC: no earlier stack ${OLDF:+$OLDF/$SIDE.tbc }- frame-count check SKIPPED"
  OL=$OLDF/${SIDE}_stack_redo.log; [ -s "$OL" ] || OL=$OLDF/${SIDE}_stack.log
  if [ ${#ins[@]} -gt 1 ] && [ -s "$OL" ]; then
    # noise normalised to each stack's json white-black range (the decoders' levels differ)
    q=$("$PY" "$B/stack_quality.py" "$OL=$OLDF/$SIDE.tbc.json" "$SL=$STKO.tbc.json" 2>&1 | tr '\n' ' ')
    log "stack quality (predicted noise; earlier then new): $q"
    B=$B "$PY" - "$OL" "$SL" "$OLDF/$SIDE.tbc.json" "$STKO.tbc.json" <<'E' || qc="predicted noise worse than the earlier stack ($q)"
import os, sys; sys.path.insert(0, os.environ['B'])
from stack_quality import predicted, level_range
j = sys.argv[3:5] if level_range(sys.argv[3]) and level_range(sys.argv[4]) else [None, None]
o, n = predicted(sys.argv[1], j[0])[0], predicted(sys.argv[2], j[1])[0]
sys.exit(0 if n <= o * 1.05 else 1)
E
  elif [ ${#ins[@]} -gt 1 ]; then
    log "stack QC: no earlier stack log${OLDF:+ ($OLDF/${SIDE}_stack_redo.log / _stack.log)} - predicted-noise check SKIPPED"
  else
    log "stack QC: single capture - predicted-noise check skipped"
  fi
  log "stack QC: $NFR frames (earlier $ON): $qc"
  [ "$qc" = ok ] || sethold "stack QC: $qc ($STKO.tbc)"
  stamp "$SCR/stackqc.ok"
fi

# ---- audio, dropout-correct --------------------------------------------------------------
# squelch_pcm.py mutes carrier-off spans (VBI + uncorrelated-L/R noise); mono_pcm.py then combines the two
# channels where the VBI says the sound is mono (less hiss, one-channel clicks repaired).  Order matters: the
# combine would make carrier-off noise identical in both channels, hiding it from the squelch.
ASQ=$OUT/${SIDE}_xm4_squelched2.pcm APCM=$OUT/${SIDE}_xm4_audio2.pcm
[ -s "$ASQ" ] || "$PY" "$LD/scripts/squelch_pcm.py" "$STKO.pcm" "$ASQ" > "$OUT/${SIDE}_xm4_squelch2.log" 2>&1
[ -s "$ASQ" ] || ASQ=$STKO.pcm
[ -s "$APCM" ] || "$PY" "$LD/scripts/mono_pcm.py" "$ASQ" "$APCM" --json="$STKO.tbc.json" --report > "$OUT/${SIDE}_xm4_mono.log" 2>&1
[ -s "$APCM" ] || { log "mono_pcm failed ($(tail -1 "$OUT/${SIDE}_xm4_mono.log" | cut -c1-200)) - squelched audio used"; APCM=$ASQ; }
MKV=$OUT/$SIDE.v8.mkv MKVOK=0
stamped "$MKV" "$MKV.stackid" && grep -q '\[newdecode\] done' "$OUT/${SIDE}_v8_decode.log" 2>/dev/null \
  && [ "$(nframes "$MKV")" = "$NFR" ] && MKVOK=1
if [ "$(stat -c %s "$SCR/doc.tbc" 2>/dev/null)" = "$(stat -c %s "$STKO.tbc")" ] && grep -q "Total concealments" "$SCR/doc.log" 2>/dev/null \
    && stamped "$SCR/doc.tbc" "$SCR/doc.tbc.stackid"; then
  log "dropout-corrected stack already in scratch, reusing"
  [ -s "$SCR/doc.tbc.stackid" ] || stamp "$SCR/doc.tbc.stackid"
elif [ $MKVOK = 0 ]; then
  log "dropout-correct"
  rm -f "$SCR/doc.tbc.stackid"
  ld-dropout-correct --input-json "$STKO.tbc.json" --output-json "$SCR/doc.tbc.json" \
      "$STKO.tbc" "$SCR/doc.tbc" > "$SCR/doc.log" 2>&1 || die "dropout-correct failed"
  stamp "$SCR/doc.tbc.stackid"
  log "$(grep -i 'Total concealments' "$SCR/doc.log" | tail -1)"
fi
if [ -e "$V7/AUTODELETE" ] && [ ${#ins[@]} -gt 1 ]; then
  for t in "${ins[@]}"; do rm -f "$t" "$t.db" "$t.json" "${t%.tbc}.pcm" "${t%.tbc}.efm"; done
  log "capture decodes deleted (stack + doc done)"
fi

# ---- model-based decode (GPU) ----------------------------------------------------------------
if [ $MKVOK = 1 ]; then
  log "$SIDE.v8.mkv already complete, reusing"
  [ -s "$MKV.stackid" ] || stamp "$MKV.stackid"
else
  [ -e "$MKV" ] && log "$SIDE.v8.mkv is not from this stack or incomplete - decoding again"
  acquire gpu "${V7_GPUS:-2}"
  log "chunked newdecode (--dropout_mask 1, master yuv422p10le) -> $MKV"
  rm -f "$MKV.stackid"
  ND_MASTER=p10 ND_CD=$ND_DIR ND_EXTRA="--dropout_mask 1" "$B/nd_chunked.sh" "$SIDE" "$SCR" "$APCM" v8 \
    || die "chunked newdecode failed"
  stamp "$MKV.stackid"
  release
fi
"$PY" - "$OUT/${SIDE}_v8_satratio.txt" <<'E' | while read -r l; do log "decode QC: $l"; done
import sys, numpy as np
try:
    r = np.loadtxt(sys.argv[1], ndmin=2)
    if r.size == 0:
        print('video-solved 0'); sys.exit()
    print('video-solved %d, satratio p5 %.3f median %.3f, fallbacks %d' % (len(r), np.percentile(r[:, 1], 5), np.median(r[:, 1]), r[:, 2].sum()))
except Exception as e:
    print('no satratio (%s)' % e)
E
log "decode: $(grep '\[newdecode\] done: chunked' "$OUT/${SIDE}_v8_decode.log" | cut -c1-300)"

# ---- PVF --------------------------------------------------------------------------------------
F= PL=$OUT/${SIDE}_v8_pvf.log d=$SLOT crf= OLDPVF=video.pvf.pre-v8 PVFID=$OUT/${SIDE}_v8_pvf.stackid
PVFMAX=${V7_PVF_MAX:-4000000000}       # bytes (FAT32 / 32-bit PVF offsets)
build_pvf(){ # <crf>
  # stillweave, make_pvfv2 and verify_pvf all check the frame count against the stack's
  # ($NFR = the master's), and stillweave's own exit status counts: a reader that died early used to
  # give a short but valid-looking PVF.
  # the FIFO on a Linux filesystem (mkfifo fails on drvfs, where $TMPDIR may point)
  local fd; fd=$(mktemp -d "${AIV_FIFO_DIR:-/tmp}/aiv-pvf.XXXXXX") || die "cannot make a FIFO directory"
  F=$fd/${SIDE}_weave.y4m
  rm -f "$d/.video.pvf.tmp"; mkfifo "$F" || { rm -rf "$fd"; die "cannot create FIFO $F"; }
  # --classes: the decoder's own still / interlaced decision per frame (nd_chunked.sh writes it with the master);
  # a missing file makes stillweave fail, so the PVF is rejected rather than built from the comb score alone
  "$PY" "$SW" "$MKV" --expect-frames "$NFR" --classes "$OUT/${SIDE}_v8_classes.txt" --log "$OUT/${SIDE}_v8_weave.log" > "$F" 2>> "$PL" &
  local wp=$! rc
  echo "=== make_pvfv2 --crf $1 ===" >> "$PL"
  nice -n 10 "$PY" "$PVT/make_pvfv2.py" "$F" "$d/.video.pvf.tmp" --dedupe --crf "$1" --probe-from "$MKV" \
      --tbc-json "$STKO.tbc.json" --audio-input "$APCM" --audio-input-rate 44100 --expect-frames "$NFR" \
      --vf "scale=832:576:flags=lanczos" >> "$PL" 2>&1 \
   && nice -n 10 "$PY" "$PVT/verify_pvf.py" "$d/.video.pvf.tmp" --expect-frames "$NFR" >> "$PL" 2>&1
  rc=$?; [ $rc = 0 ] || kill $wp 2>/dev/null
  if ! wait $wp && [ $rc = 0 ]; then echo "stillweave failed - PVF rejected" >> "$PL"; rc=1; fi
  rm -rf "$fd"; return $rc
}
# the slot's video.pvf is one this v8 chain installed (replaced in place on a rebuild) or an older one
# (kept as $OLDPVF, which must not exist yet)
V8CUR=0; { grep -q "PVF v8 OK" <(grep "AIV $SIDE " "$LOG") || [ -s "$PVFID" ]; } && V8CUR=1
if grep -q "PVF v8 OK" <(grep "AIV $SIDE " "$LOG") && stamped "$d/video.pvf" "$PVFID"; then
  log "PVF already installed"
  [ -s "$PVFID" ] || stamp "$PVFID"
else
  [ $V8CUR = 1 ] && log "installed v8 PVF is not from this stack - rebuilding it (replaced in place, $OLDPVF untouched)"
  # checked before the hours-long build, not after it
  for n in $(echo "$SLOTS" | tr ',' ' '); do
    e=$V7_SLOTROOT/BeebVFS$n
    [ $V8CUR = 0 ] && [ -e "$e/video.pvf" ] && [ -e "$e/$OLDPVF" ] && die "$e/$OLDPVF already exists - not overwriting (no PVF built)"
  done
  acquire pvf 1; mkdir -p "$d"
  # CRF 17, stepping up (18, 19, 20) while the PVF is >= 4 GB (FAT32 / PVF offsets)
  for c in 17 18 19 20; do
    log "PVF v8 (still-weave, CRF $c, dedupe) -> $d"
    if build_pvf $c; then
      sz=$(stat -c%s "$d/.video.pvf.tmp")
      [ "$sz" -lt "$PVFMAX" ] && { crf=$c; break; }
      log "CRF $c PVF is $sz bytes (>= $PVFMAX)"
    elif awk '/^=== make_pvfv2/ {s = ""} {s = s $0 "\n"} END {printf "%s", s}' "$PL" | grep -q 'exceeds 4GB'; then
      log "CRF $c PVF exceeds the 4 GB PVF limit"
    elif [ $c = 17 ]; then
      log "CRF 17 PVF failed (see $PL) - trying CRF 18"
    else
      rm -f "$d/.video.pvf.tmp"; die "PVF v8 failed at CRF $c (see $PL)"
    fi
  done
  [ -n "$crf" ] || { rm -f "$d/.video.pvf.tmp"; die "PVF v8 still >= $PVFMAX bytes at CRF 20 (see $PL)"; }
  release
  kept="no previous PVF"
  [ $V8CUR = 1 ] && [ -e "$d/video.pvf" ] && kept="replaced the earlier v8 PVF ($OLDPVF untouched)"
  if [ -e "$d/video.pvf" ] && [ $V8CUR = 0 ]; then
    [ -e "$d/$OLDPVF" ] && die "$d/$OLDPVF already exists - not overwriting (new PVF left at .video.pvf.tmp)"
    mv -f "$d/video.pvf" "$d/$OLDPVF"; kept="previous kept as $OLDPVF"
  fi
  mv -f "$d/.video.pvf.tmp" "$d/video.pvf"; stamp "$PVFID"
  log "PVF v8 OK (CRF $crf) $(stat -c%s "$d/video.pvf") bytes; $(grep '^dedupe: .*share' "$PL" | tail -1); $kept"
  for n in $(echo "$SLOTS" | tr ',' ' ' | cut -d' ' -f2-); do          # further slots get a copy
    [ "$n" = "${SLOTS%%,*}" ] && continue
    e=$V7_SLOTROOT/BeebVFS$n; mkdir -p "$e"
    cp "$d/video.pvf" "$e/.video.pvf.tmp" && cmp -s "$d/video.pvf" "$e/.video.pvf.tmp" || die "copy to $e failed"
    o=$OLDPVF
    k="no previous PVF"; [ -e "$e/video.pvf" ] && k="replaced the earlier v8 copy"
    if [ -e "$e/video.pvf" ] && [ $V8CUR = 0 ]; then [ -e "$e/$o" ] && die "$e/$o exists"; mv -f "$e/video.pvf" "$e/$o"; k="previous kept as $o"; fi
    mv -f "$e/.video.pvf.tmp" "$e/video.pvf"; log "PVF copied to $(basename "$e") ($k)"
  done
fi
rm -rf "$SCR/doc.tbc" "$SCR/doc.tbc.json" "$SCR/doc.tbc.stackid" "$SCR/ndc"

# ---- per-frame quality record (quality/quality_db.py).  Holds no slot; read-only on the stack, master and PVF.
# "done" only when the core steps (run, fields, PVF) worked; otherwise the failed steps are
# logged and the record is retried on the next run of this side.
FQ=$DAR/quality QL=$OUT/${SIDE}_v8_quality.log
if ! { grep -q "quality record done" <(grep "AIV $SIDE " "$LOG") && stamped "$QL" "$QL.stackid"; }; then
  log "quality record: stack frames, blank-line noise + VITS, newdecode, PVF encode loss"
  qf= qn= RUN=
  [ -n "$crf" ] || crf=$(grep -o "AIV $SIDE PVF v8 OK (CRF [0-9]*" "$LOG" | tail -1 | grep -o '[0-9]*$')
  {
    RUN=$("$PY" $FQ/quality_db.py run "$SIDE" v8 "ld=$(git -C "$LD" log --oneline -1 | cut -c1-8)" \
          "ld_opts=$LD_OPTS" "stacker=$(md5sum "$STK" | cut -c1-8)" "nd=$(tr ' ' : < "$OUT/${SIDE}_v8_decoder.txt" 2>/dev/null || echo "$ND_DIR:?")" \
          master=yuv422p10le "captures=${#ins[@]}" "pvf_crf=$crf") && [ -n "$RUN" ] || qf="$qf run"
    if [ -n "$RUN" ]; then
      # (no TSV, e.g. a side rebuilt from a single capture: clears the run's stack rows from an earlier build)
      "$PY" $FQ/quality_db.py stack "$RUN" "$OUT/${SIDE}_stack_frames.tsv" || qn="$qn stack"
      { nice -n 10 "$PY" $FQ/fq_side.py "$STKO.tbc" "$OUT/${SIDE}_v8_fields.csv" --vits \
        && "$PY" $FQ/quality_db.py fields "$RUN" "$OUT/${SIDE}_v8_fields.csv"; } || qf="$qf fields"
      "$PY" $FQ/quality_db.py nd "$RUN" "$OUT/${SIDE}_v8_satratio.txt" "$OUT/${SIDE}_v8_classes.txt" || qn="$qn nd"
      { nice -n 10 $FQ/pvf_quality.sh "$SLOT/video.pvf" "$MKV" "$OUT/${SIDE}_v8_pvfq" "$PL" "$OUT/${SIDE}_v8_classes.txt" \
        && "$PY" $FQ/quality_db.py pvf "$RUN" "$OUT/${SIDE}_v8_pvfq"; } || qf="$qf pvf"
      "$PY" $FQ/quality_db.py summary "$RUN" || qn="$qn summary"
    fi
  } > "$QL" 2>&1
  if [ -z "$qf" ]; then
    stamp "$QL.stackid"
    log "quality record done${qn:+ (non-core steps failed:$qn)}: $(tail -1 "$QL" | cut -c1-400)"
  else
    log "quality record FAILED (steps:$qf${qn:+; also$qn}; see $QL) - retried on the next run of this side"
  fi
fi
log "=== v8 done: $MKV + $(basename "$SLOT")/video.pvf ==="
