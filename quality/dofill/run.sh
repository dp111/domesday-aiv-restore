#!/bin/bash
# Dropout-fill comparison through newdecode (see prep.py).  A copy/no mask, B interp/no mask,
# C copy/mask (production), D interp/mask.
#   run.sh <benchdir>  (a make_bench.py benchmark, after prep.py)
B=${1:?benchdir}
cd "$(dirname "$0")/.."
for v in "dfC copy 1" "dfA copy 0" "dfD avg 1" "dfB avg 0"; do set -- $v
  echo "$(date +%T) $1 start"
  DOC_FROM=dofill/$2 STEPS=nd ND_EXTRA="--dropout_mask $3" ./run_bench.sh $B $1
  echo "$(date +%T) $1 done"
done
