#!/bin/bash
# GPU / memory gate for the pipeline (run beside run.sh).  A side holds its GPU lock through the concat +
# ffprobe verify of its ~100 GB MKV, which took 9 h on a busy HDD while the GPU sat idle.  So both GPU
# lock slots are usable, and this gate keeps it to one newdecode at a time (each peaks ~32 GB):
# PAUSE_gpu (honoured by acquire) while an nd_gpuwrap runs or a gpu lock was taken in the last 5 minutes
# (its newdecode may not have started yet).
. "$(dirname "${BASH_SOURCE[0]}")/conf.sh"
last=
while :; do
  busy=
  pgrep -f '[n]d_gpuwrap.py' > /dev/null && busy=newdecode
  # nd_chunked between chunks counts too (10-22 21:29 North slipped in), but not once it is joining / checking
  # its chunks (concat + ffprobe, CPU/disk only: 10-24 the GPU sat idle through EcoDisc_S1's)
  # (the final .mkv.tmp.mkv check only, not the per-chunk ffprobe of chunk_NN.mkv)
  [ -z "$busy" ] && pgrep -f '[n]d_chunked.sh' > /dev/null && ! pgrep -f '[f]fmpeg .*-f concat|[f]fprobe .*\.mkv\.tmp\.mkv' > /dev/null && busy=newdecode
  for o in "$V7"/locks/gpu.*/owner; do
    [ -e "$o" ] && [ $(( $(date +%s) - $(stat -c %Y "$o") )) -lt 300 ] && busy=${busy:-new lock}
  done
  # 2026-10-23: also hold new stacks (16 jobs, many GB) while a newdecode runs: NationalB + Countryside_S1 stacks
  # beside North filled RAM + swap (ENOMEM in North, South PVF, capture decodes)
  if [ -n "$busy" ]; then touch "$V7/PAUSE_gpu" "$V7/PAUSE_stack"; else rm -f "$V7/PAUSE_gpu" "$V7/PAUSE_stack"; fi
  [ "$busy" != "$last" ] && v7log "gpugate: ${busy:-GPU free}"
  last=$busy
  # 2026-10-23: a lock whose owner process is gone (South's PVF rebuild died holding pvf.1; North's PVF waited 7.5 h)
  for o in "$V7"/locks/*/owner; do
    [ -e "$o" ] || continue
    pid=$(awk '{print $2}' "$o"); [ -n "$pid" ] && ! kill -0 "$pid" 2> /dev/null \
      && [ $(( $(date +%s) - $(stat -c %Y "$o") )) -gt 120 ] && { rm -f "$o"; rmdir "${o%/owner}"; v7log "gpugate: stale lock ${o%/owner} removed"; }
  done
  sleep 10
done
