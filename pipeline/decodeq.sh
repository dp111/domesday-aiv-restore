#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Capture-decode queue: every capture of every side in sides.conf, in side order, V7_WIDE (4) at a time,
# so the next side's captures decode while the previous side stacks.  Each decode goes to the drive with
# most room (config V7_DRIVES) and is announced to its side as $CAPS/<side>/<cap>.path (tbc base) +
# .done / .failed.  A capture that does not fit anywhere waits for space (sides free theirs after stacking).
# Disk accounting counts only decodes that are still running (<cap>.pid = the decode's shell, or else an
# ld-decode / db2json process on its output), and only the part not yet written (<cap>.need minus the
# current .tbc size): a dead partial decode is already in df.  A capture whose decode is still running
# (e.g. this queue was restarted) is not started again, and counts towards V7_WIDE.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/conf.sh"
WIDE=${V7_WIDE:-4}
mkdir -p "$CAPS"
log(){ v7log "decodeq $*"; }

pick_dir(){ # <bytes needed> -> dir with most room above reserve, or nothing
  local need=$1 best= bestfree=0 d dir mnt res free
  for d in $V7_DRIVES; do
    IFS=: read -r dir mnt res <<< "$d"
    free=$(df --output=avail -B1 "$mnt" | tail -1); free=$((free - res * 1000000000 - PENDING_${mnt//[^A-Za-z0-9_]/_}))
    [ "$free" -gt "$need" ] && [ "$free" -gt "$bestfree" ] && { best=$dir; bestfree=$free; }
  done
  echo "$best"
}

alive(){ # <capture record base>: its decode is still running
  local pid dd
  pid=$(cat "$1.pid" 2> /dev/null)
  [ -n "$pid" ] && kill -0 "$pid" 2> /dev/null && grep -aq decodeq "/proc/$pid/cmdline" 2> /dev/null && return 0
  dd=$(cat "$1.path" 2> /dev/null); [ -n "$dd" ] || return 1
  ps -eo args | awk -v d="$dd" '/ld-decode|db2json/ {for (i = 2; i <= NF; i++) if ($i == d || $i == d ".tbc.db") f = 1} END {exit !f}'
}
nalive(){ local p q k=0
  for p in "$CAPS"/*/*.path; do
    [ -e "$p" ] || continue; q=${p%.path}; [ -e "$q.done" ] || [ -e "$q.failed" ] && continue
    alive "$q" && k=$((k + 1))
  done; echo $k; }

pending(){ # <need of a record without .need>: PENDING_<mount> = what the running decodes have yet to write there
  local d m p q dd r dir v
  for d in $V7_DRIVES; do IFS=: read -r _ m _ <<< "$d"; declare -g "PENDING_${m//[^A-Za-z0-9_]/_}=0"; done
  for p in "$CAPS"/*/*.path; do
    [ -e "$p" ] || continue; q=${p%.path}; [ -e "$q.done" ] || [ -e "$q.failed" ] && continue
    alive "$q" || continue
    dd=$(cat "$p"); r=$(( $(cat "$q.need" 2> /dev/null || echo "$1") - $(stat -c %s "$dd.tbc" 2> /dev/null || echo 0) ))
    [ $r -gt 0 ] || r=0
    for d in $V7_DRIVES; do IFS=: read -r dir m _ <<< "$d"
      [[ $dd == $dir/* ]] && { v=PENDING_${m//[^A-Za-z0-9_]/_}; declare -g "$v=$(( ${!v} + r ))"; }; done
  done
}

decode(){ # <side> <ldf> <cap> <dir>
  local s=$1 ldf=$2 c=$3 w=$4/${1}_$3 n want spur= ca=$CAPS/$1/$3
  echo "$BASHPID" > "$ca.pid"
  [ "$(spur_of "$ldf")" = 1 ] && spur=--V4300D_coherent_subtract
  want=$(( $(ldf_frames "$ldf") * FB / 100 * $([[ $ldf == *_CLV_* ]] && echo 93 || echo 97) ))
  rm -f "$w.tbc" "$w.tbc.json" "$w.tbc.db" "$w.pcm" "$w.efm"
  log "$s $c start -> $4 ($(basename "$ldf"), spur ${spur:+subtract}${spur:-none})"
  ( cd "$LD" && LDDECODE_EFM_GEARSHIFT=0 "$PY" ld-decode --PAL -t 4 $LD_OPTS $spur --noEFM "$ldf" "$w" > "$w.declog" 2>&1 )
  n=$(stat -c %s "$w.tbc" 2>/dev/null || echo 0)
  if [ "$n" -lt 1000000000 ]; then            # unreadable start spins at frame 0: retry past it
    log "$s $c short ($n bytes) - retry --seek 350"
    ( cd "$LD" && LDDECODE_EFM_GEARSHIFT=0 "$PY" ld-decode --PAL -t 4 $LD_OPTS --seek 350 $spur --noEFM "$ldf" "$w" > "$w.declog" 2>&1 )
    n=$(stat -c %s "$w.tbc" 2>/dev/null || echo 0)
  fi
  # rc is 0 even when numba crashes, so judge by the output size
  if [ "$n" -lt "$want" ]; then log "$s $c FAILED ($n bytes, want >= $want; see $w.declog)"; touch "$ca.failed"; rm -f "$ca.pid"; return 1; fi
  "$PY" "$LD/scripts/db2json.py" "$w.tbc.db" > /dev/null 2>&1
  [ -s "$w.tbc.json" ] || { log "$s $c: db2json failed"; touch "$ca.failed"; rm -f "$ca.pid"; return 1; }
  touch "$ca.done"; rm -f "$ca.pid"; log "$s $c done ($n bytes)"
}

log "start ($WIDE wide; ld-decode $(git -C "$LD" log --oneline -1 | cut -c1-8) $LD_OPTS)"
for s in $(echo "$V7_SIDES" | awk 'NF {print $2}'); do
  mkdir -p "$CAPS/$s"
  for ldf in $(side_ldfs "$s"); do
    c=$(cap_of "$ldf"); ca=$CAPS/$s/$c
    [ -e "$ca.done" ] && continue
    if alive "$ca"; then log "$s $c: a decode of it is still running - not started again"; continue; fi
    rm -f "$ca.failed"
    while [ "$(nalive)" -ge "$WIDE" ] || [ -e "$V7/PAUSE_DECODE" ] \
          || [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done
    need=$(( $(ldf_frames "$ldf") * FB * 105 / 100 ))
    pending "$need"
    dir=$(pick_dir "$need"); waited=0
    while [ -z "$dir" ]; do
      [ $waited = 0 ] && log "$s $c waits for disk space ($((need / 1000000000)) GB; running decodes still to write:$(for d in $V7_DRIVES; do IFS=: read -r _ m _ <<< "$d"; v=PENDING_${m//[^A-Za-z0-9_]/_}; echo -n " $m $((${!v} / 1000000000)) GB"; done))"
      waited=1; sleep 300; pending "$need"; dir=$(pick_dir "$need")
    done
    mkdir -p "$dir"; echo "$dir/${s}_$c" > "$ca.path"; echo "$need" > "$ca.need"
    decode "$s" "$ldf" "$c" "$dir" &
    sleep 30
  done
  log "$s: all captures queued"
done
wait
log "all decodes done"
