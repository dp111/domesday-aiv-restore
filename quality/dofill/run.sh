#!/bin/bash
# Dropout-fill comparison through newdecode (see prep.py).  A copy/no mask, B interp/no mask,
# C copy/mask (production), D interp/mask.
#   run.sh <benchdir>  (a make_bench.py benchmark)
# Makes what it needs: per segment, prep.py's dofill/{inj,avg}.tbc, dofill/copy.tbc (ld-dropout-correct
# of inj) and the doc.tbc links run_bench.sh's DOC_FROM=dofill/<copy|avg> reads (dofill/<v>/doc.tbc[.json]).
set -u
B=${1:?benchdir}
D=$(cd "$(dirname "$0")" && pwd)
. "$D/../../config.sh"
[ -d "$B" ] || { echo "no benchmark dir $B"; exit 1; }
for s in $(ls -d "$B"/seg${SEGS:-*}/ "$B"/pat${SEGS:-*}/ 2>/dev/null); do
  s=${s%/}; o=$s/dofill
  if [ ! -s "$o/inj.tbc" ] || [ ! -s "$o/avg.tbc" ] || [ ! -s "$o/mask.npy" ]; then
    echo "$(date +%T) $s: prep"
    "$PY" "$D/prep.py" "$s" || { echo "$s: prep.py failed"; exit 1; }
  fi
  if [ ! -s "$o/copy.tbc" ] || ! grep -q "Total concealments" "$o/copy.log" 2>/dev/null; then
    ld-dropout-correct --input-json "$o/inj.tbc.json" --output-json "$o/copy.tbc.json" "$o/inj.tbc" "$o/copy.tbc" \
      > "$o/copy.log" 2>&1 || { echo "$s: ld-dropout-correct failed (see $o/copy.log)"; exit 1; }
  fi
  for v in copy avg; do
    mkdir -p "$o/$v"
    ln -sfn "../$v.tbc" "$o/$v/doc.tbc"; ln -sfn "../$v.tbc.json" "$o/$v/doc.tbc.json"
  done
done
cd "$D/.."
for v in "dfC copy 1" "dfA copy 0" "dfD avg 1" "dfB avg 0"; do set -- $v
  echo "$(date +%T) $1 start"
  DOC_FROM=dofill/$2 STEPS=nd ND_EXTRA="--dropout_mask $3" ./run_bench.sh "$B" $1
  echo "$(date +%T) $1 done"
done
