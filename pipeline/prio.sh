#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# CPU scheduling for the pipeline (run beside run.sh).  On the i9-14900K (8 P + 16 E cores, 32 threads) the CPU is the bottleneck.
#  - nice: stacks 5, capture ld-decodes 15 (GPU newdecode stays 0)
#  - while a stack runs, only the V7_STACK_DECODES (2) oldest capture decodes run; the others are
#    SIGSTOPped and resumed when no stack runs.  Measured 10-20 02:00-03:10 (NationalA 13-cap stack
#    + NationalB CLV decodes): 4 decodes -> stack 2640 fr/h + decodes 22.5k fr/h; 2 decodes -> stack
#    3340 + decodes 25.5k; stack alone 4262.
# 2026-10-20: while a GPU newdecode runs (peaks ~31 GB), at most V7_ND_DECODES (4) capture decodes run.
# 2026-10-21: one HDD-heavy job (ld-dropout-correct / ffmpeg concat) at a time; 3 together got ~12 MB/s.
# Linux nice is per thread: every thread of every matching process and its descendants is reniced.
# /proc comm is cut to 15 characters ("ld-dropout-corr"), so dropout-correct is matched by prefix.  On exit
# (TERM / INT / HUP too) every process this scheduler stopped is sent SIGCONT, so stopping it never leaves
# decodes frozen.
. "$(dirname "${BASH_SOURCE[0]}")/conf.sh"
NDEC=${V7_STACK_DECODES:-2}
tree(){ local p; for p in "$@"; do echo $p; tree $(pgrep -P $p); done; }
setn(){ local n=$1 p t; shift
  for p in $(tree $(pgrep -f "$1")); do for t in /proc/$p/task/*; do
    [ "$(awk '{print $19}' $t/stat 2>/dev/null)" = "$n" ] || renice -n $n -p ${t##*/} > /dev/null 2>&1; done; done; }
STOPPED=                                                           # pids this scheduler has SIGSTOPped
resume_all(){ local p; for p in $STOPPED; do kill -CONT $(tree $p) 2> /dev/null; done; STOPPED=; }
trap 'resume_all' EXIT
# (HUP: started under nohup, HUP is ignored and bash cannot trap it; stop the scheduler with TERM)
trap 'v7log "prio: stopped by signal - resuming all paused processes"; exit 0' TERM INT HUP
last=
while :; do
  setn 15 '[l]d-decode --PAL'
  setn 5 '[l]ddecode.stack'
  decs=$(ps -eo pid,etimes,args --sort=-etimes | awk '/[l]d-decode --PAL/ {print $1}')   # oldest first
  # 2026-10-24: a chunk concat (~110 GB S: -> S:) ran at ~2.5 MB/s beside 4 capture decodes reading .ldf from S:
  if pgrep -f "[f]fmpeg .*-f concat" > /dev/null; then keep=0
  elif pgrep -f "[l]ddecode.stack" > /dev/null; then keep=$NDEC
  elif pgrep -f '[n]d_gpuwrap.py' > /dev/null; then keep=${V7_ND_DECODES:-4}   # newdecode: memory + its CPU thread
  else keep=99; fi
  i=0; st=
  for p in $decs; do
    i=$((i + 1))
    if [ $i -le $keep ]; then kill -CONT $(tree $p) 2> /dev/null; else kill -STOP $(tree $p) 2> /dev/null; st="$st $p"; STOPPED="$STOPPED $p"; fi
  done
  # S: and T: are single HDDs: run one big sequential S: job at a time (dropout-correct, chunk concat), oldest first
  i=0
  # 2026-10-24: only the real ld-dropout-correct / ffmpeg processes (not shells whose command text mentions
  # them) and only on single hard disks (config V7_HDD_RE: a regex of their paths; empty = no I/O gating)
  for p in $(ps -eo pid,etimes,comm,args --sort=-etimes | awk -v hdd="${V7_HDD_RE:-^$}" '$3 ~ /^ld-dropout-co/ && $0 ~ hdd {print $1} $3 == "ffmpeg" && /-f concat/ && $0 ~ hdd {print $1}'); do
    i=$((i + 1)); if [ $i = 1 ]; then kill -CONT $p 2> /dev/null; else kill -STOP $p 2> /dev/null; st="$st io:$p"; STOPPED="$STOPPED $p"; fi
  done
  [ "$st" != "$last" ] && v7log "prio: keep $keep - capture decodes paused:${st:- none}"
  last=$st
  STOPPED=$(echo "$STOPPED" | tr ' ' '\n' | awk 'NF && !s[$0]++' | while read -r p; do kill -0 "$p" 2> /dev/null && echo "$p"; done | tr '\n' ' ')   # drop exited pids
  sleep 60 & wait $!                                                # (a signal is handled at once, not after the sleep)
done
