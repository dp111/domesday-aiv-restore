# SPDX-License-Identifier: GPL-3.0-or-later
# Pipeline settings and helpers, sourced by run.sh, decodeq.sh, side.sh, prio.sh and gpugate.sh.
B=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)             # this directory
. "$B/../config.sh"
. "$B/sides.conf"                                          # V7_SIDES: the side table
V7=$WORK/state                     # locks/, wait/, PAUSE_*, HOLD.<side>, AUTODELETE
CAPS=$V7/caps                      # capture records: <side>/<cap>.path (tbc base) + .done / .failed
LOG=$WORK/pipeline.log
export ND_LOG=$LOG
mkdir -p "$V7" "$CAPS"
# ld-decode options for every capture decode: VITS white-level luma gain (VITS A/B 74 -> 42 failed checks)
LD_OPTS=${LD_OPTS:---vits_white_level}
FB=$((1135 * 313 * 2 * 2))          # bytes per PAL frame in a .tbc

side_field(){ echo "$V7_SIDES" | awk -v s="$1" -v f="$2" '$2 == s {print $f}'; }
side_ldfs(){ ls $(side_field "$1" 5) 2>/dev/null; }
# capture id from the file name: dsN from DD86-DSN, or the UN _DiscN_.  NationalA DS1 was captured in
# two parts (00001-37480 and 39000-54000): ds1a / ds1b.
cap_of(){ local b; b=$(basename "$1")
  echo "$b" | grep -o 'DD86-DS[0-9]*\|_Disc[0-9]*_' | sed 's/DD86-DS/ds/; s/_Disc\([0-9]*\)_/ds\1/' | {
    read -r c
    if [ "$c" = ds1 ] && [[ $b == Domesday_DD86-DS1_NationalA_* ]]; then
      [[ $b == *_00001-* ]] && c=ds1a || c=ds1b
    fi
    echo "$c"; }; }
# 1 if the capture has the LD-V4300D ~8.47 MHz spur (spur_probe.txt, made by probe_v4300_spur.py):
# decode it with --V4300D_coherent_subtract.  UN .raw.oga: all 7 probed 17-23 dB.
spur_of(){
  [[ $1 == *.raw.oga ]] && { echo 1; return; }
  grep -h "^$(basename "$1") " "$B/spur_probe.txt" 2>/dev/null | grep -q 'SPUR$' && echo 1 || echo 0; }
ldf_frames(){ # nominal frames from the capture name: CAV 00001-54000, CLV 00-60 (minutes)
  [[ $1 == *.raw.oga ]] && { echo 42000; return; }      # UN: ~42580-42800 frames
  local r; r=$(basename "$1" .ldf | grep -o '[0-9]*-[0-9]*$')
  if [[ $1 == *_CLV_* ]]; then echo $(( (10#${r#*-} - 10#${r%-*}) * 60 * 25 )); else echo $(( 10#${r#*-} - 10#${r%-*} + 1 )); fi; }
v7log(){ echo "$(date '+%m-%d %H:%M:%S') AIV $*" >> "$LOG"; }
if [ -n "${V7_TEST:-}" ]; then . "$V7_TEST"; fi     # test overrides (side table, $V7, log)
