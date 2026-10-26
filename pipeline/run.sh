#!/bin/bash
# Start the whole pipeline (conf.sh, sides.conf): the decode queue + one side.sh per side; the sides wait
# for their captures and for stack / GPU / PVF slots.  Run prio.sh and gpugate.sh beside it.
. "$(dirname "${BASH_SOURCE[0]}")/conf.sh"
mkdir -p "$V7/out" "$V7_OUT" "$V7_SCRROOT" "$CAPS"
cd "$B"
setsid nohup ./decodeq.sh > "$V7/out/decodeq.out" 2>&1 < /dev/null &
for s in $(echo "$V7_SIDES" | awk 'NF {print $2}'); do
  setsid nohup ./side.sh "$s" > "$V7/out/$s.out" 2>&1 < /dev/null &
  sleep 1
done
v7log "run.sh: launched decode queue + $(echo "$V7_SIDES" | awk 'NF' | wc -l) side chains"
