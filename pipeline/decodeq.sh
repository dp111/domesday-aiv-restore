#!/bin/bash
# Capture-decode queue: every capture of every side in sides.conf, in side order, V7_WIDE (4) at a time,
# so the next side's captures decode while the previous side stacks.  Each decode goes to the drive with
# most room (config V7_DRIVES) and is announced to its side as $CAPS/<side>/<cap>.path (tbc base) +
# .done / .failed.  A capture that does not fit anywhere waits for space (sides free theirs after stacking).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/conf.sh"
WIDE=${V7_WIDE:-4}
mkdir -p "$CAPS"
log(){ v7log "decodeq $*"; }

pick_dir(){ # <bytes needed> -> dir with most room above reserve, or nothing
  local need=$1 best= bestfree=0 d dir mnt res free
  for d in $V7_DRIVES; do
    IFS=: read -r dir mnt res <<< "$d"
    free=$(df --output=avail -B1 "$mnt" | tail -1); free=$((free - res * 1000000000 - PENDING_${mnt//\//_}))
    [ "$free" -gt "$need" ] && [ "$free" -gt "$bestfree" ] && { best=$dir; bestfree=$free; }
  done
  echo "$best"
}

decode(){ # <side> <ldf> <cap> <dir>
  local s=$1 ldf=$2 c=$3 w=$4/${1}_$3 n want spur= ca=$CAPS/$1/$3
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
  if [ "$n" -lt "$want" ]; then log "$s $c FAILED ($n bytes, want >= $want; see $w.declog)"; touch "$ca.failed"; return 1; fi
  "$PY" "$LD/scripts/db2json.py" "$w.tbc.db" > /dev/null 2>&1
  [ -s "$w.tbc.json" ] || { log "$s $c: db2json failed"; touch "$ca.failed"; return 1; }
  touch "$ca.done"; log "$s $c done ($n bytes)"
}

log "start ($WIDE wide; ld-decode $(git -C "$LD" log --oneline -1 | cut -c1-8) $LD_OPTS)"
for s in $(echo "$V7_SIDES" | awk 'NF {print $2}'); do
  mkdir -p "$CAPS/$s"
  for ldf in $(side_ldfs "$s"); do
    c=$(cap_of "$ldf"); ca=$CAPS/$s/$c
    [ -e "$ca.done" ] && continue
    rm -f "$ca.failed"
    while [ "$(jobs -rp | wc -l)" -ge "$WIDE" ] || [ -e "$V7/PAUSE_DECODE" ] \
          || [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done
    need=$(( $(ldf_frames "$ldf") * FB * 105 / 100 ))
    # room for the decodes already running counts against each drive (they are still growing)
    for d in $V7_DRIVES; do IFS=: read -r _ m _ <<< "$d"; declare "PENDING_${m//\//_}=0"; done
    for p in "$CAPS"/*/*.path; do
      [ -e "$p" ] || continue; q=${p%.path}; [ -e "$q.done" ] || [ -e "$q.failed" ] && continue
      dd=$(cat "$p"); for d in $V7_DRIVES; do IFS=: read -r dir m _ <<< "$d"
        [[ $dd == $dir/* ]] && { v=PENDING_${m//\//_}; declare "$v=$(( ${!v} + need ))"; }; done
    done
    dir=$(pick_dir "$need"); waited=0
    while [ -z "$dir" ]; do
      [ $waited = 0 ] && log "$s $c waits for disk space ($((need / 1000000000)) GB)"
      waited=1; sleep 300; dir=$(pick_dir "$need")
    done
    mkdir -p "$dir"; echo "$dir/${s}_$c" > "$ca.path"
    decode "$s" "$ldf" "$c" "$dir" &
    sleep 30
  done
  log "$s: all captures queued"
done
wait
log "all decodes done"
