#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Run the production stack -> dropout-correct -> newdecode chain on an fq benchmark (make_bench.py output).
#   run_bench.sh <benchdir> <label> [caps...]
# Output per segment: <seg>/<label>/{stack.tbc,doc.tbc,out.yuv,*.log}; then score.py scores it.
# The chain mirrors pipeline/side.sh + nd_chunked.sh (same flags) unless overridden:
#   STK        ld-decode tree whose lddecode/stack.py stacks (default $LD, production)
#   STACK_EXTRA extra stack.py options      (production: --cross_master, auto master clustering)
#   ND_CD      decoder dir                 (default newdecode/, production)
#   ND_EXTRA   newdecode options           (default "--dropout_mask 1", production)
#   STEPS      subset of "stack doc nd"    (default all three)
#   NOGPU=1    stop after doc (no newdecode)
#   ND_WRAP    newdecode wrapper (default pipeline/nd_gpuwrap.py; bench runs use a renamed copy, since the
#              production gpugate pauses stacks while any nd_gpuwrap.py runs)
#   DOC_FROM   decode another label's doc.tbc (e.g. DOC_FROM=prod STEPS=nd), or DOC_FROM=clean for the
#              noise-free composite (the decoder's own ceiling)
# caps: capture names to use (default all synthetic captures), e.g. "ds1 ds2 ds4" for a reduced stack, or a
# single capture for a single-copy side (then the stack step is skipped and that capture is used directly).
set -u
B=$1 LAB=$2; shift 2
CAPS=("$@")
. "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
STK=${STK:-$LD}
ND_CD=${ND_CD:-$DAR/newdecode}
ND_EXTRA=${ND_EXTRA---dropout_mask 1}
STEPS=${STEPS:-stack doc nd}
export ND_FORCE_T3=1
for d in $(ls -d "$B"/seg${SEGS:-*}/ "$B"/pat${SEGS:-*}/ 2>/dev/null); do   # SEGS=03000 for one segment
  d=${d%/}; o=$d/$LAB; mkdir -p "$o"
  ins=()
  if [ ${#CAPS[@]} = 0 ]; then ins=("$d"/syn_*.tbc); else for c in "${CAPS[@]}"; do ins+=("$d/syn_$c.tbc"); done; fi
  {
    echo "label $LAB  stk $STK $(md5sum "$STK/lddecode/stack.py" | cut -c1-8)  stack_extra ${STACK_EXTRA:-}  nd $ND_CD $(md5sum "$ND_CD/newdecode.py" 2>/dev/null | cut -c1-8) $ND_EXTRA  caps ${CAPS[*]:-all}"
  } > "$o/run.txt"
  src=$o/stack
  if [ ${#ins[@]} = 1 ]; then
    src=${ins[0]%.tbc}
  elif [[ " $STEPS " == *" stack "* ]]; then
    ( cd "$STK" && nice -n 5 "$PY" -m lddecode.stack --PAL --jobs ${JOBS:-6} --cross_master ${STACK_EXTRA:-} \
        -o "$o/stack" "${ins[@]}" ) > "$o/stack.log" 2>&1 || { echo "$d: stack failed"; continue; }
  fi
  if [[ " $STEPS " == *" doc "* ]]; then
    rm -f "$o/doc.tbc" "$o/doc.tbc.json"
    ld-dropout-correct --input-json "$src.tbc.json" --output-json "$o/doc.tbc.json" "$src.tbc" "$o/doc.tbc" > "$o/doc.log" 2>&1 \
      || { echo "$d: dropout-correct failed"; continue; }
  fi
  [ "${NOGPU:-0}" = 1 ] && { echo "$d: done (no newdecode)"; continue; }
  if [[ " $STEPS " == *" nd "* ]]; then
    rm -rf "$o/nd"
    if [ -n "${DOC_FROM:-}" ]; then
      if [ "$DOC_FROM" = clean ]; then dsrc=$d/clean; else dsrc=$d/$DOC_FROM/doc; fi
      ln -sf "$dsrc.tbc" "$o/doc.tbc"; ln -sf "$dsrc.tbc.json" "$o/doc.tbc.json"
    fi
    if [[ $ND_CD == chroma:* ]]; then     # baseline: upstream ld-chroma-decoder, e.g. ND_CD=chroma:transform3d
      ld-chroma-decoder -q -f "${ND_CD#chroma:}" -p yuv --input-json "$o/doc.tbc.json" "$o/doc.tbc" "$o/out.yuv" 2> "$o/nd.log" \
        || { echo "$d: ld-chroma-decoder failed"; continue; }
      echo "$d: done"; continue
    fi
    "$MLPY" "${ND_WRAP:-$DAR/pipeline/nd_gpuwrap.py}" "$ND_CD/newdecode.py" "$o/doc.tbc" - --scratch "$o/nd" --video 1 $ND_EXTRA 2> "$o/nd.log" \
      | ffmpeg -hide_banner -loglevel error -y -i - -f rawvideo -pix_fmt yuv444p16le "$o/out.yuv" \
      || { echo "$d: newdecode failed"; continue; }
    rm -rf "$o/nd"
  fi
  echo "$d: done"
done
