# SPDX-License-Identifier: GPL-3.0-or-later
# Example config.local.sh: the machine the 2026 restoration ran on (WSL2, i9-14900K, 48 GB, one GPU).
# S: and T: are single hard disks, E: is NVMe (it also holds the .ldf captures, so decodes go to a
# scratch directory beside them, never into the capture directories).
PY=/mnt/s/domsday/ldenv/bin/python
MLPY=/root/mlenv/bin/python
LD=/mnt/s/domsday/ld-next          # worktree on branch domesday-aiv
WORK=/mnt/s/domsday/aiv-work
V7_OUT=/mnt/s/domsday/aiv_final/v8
V7_SCRROOT=/mnt/s/domsday/v8_scratch
V7_SLOTROOT=/mnt/t/Pi1MHz
V7_OLDF=/mnt/t/aiv_video/final
V7_DRIVES="/mnt/s/domsday/v7_scratch/work:/mnt/s:600 /mnt/e/domsday_scratch/v7work:/mnt/e:100 /mnt/t/v7work:/mnt/t:200"
CAP_AIV=/mnt/e/domesday
CAP_DISCS=/mnt/s
CAP_UN=/root/v7links
V7_HDD_RE=/mnt/[st]/                # S: and T: are hard disks
ND_GPULOCK=/run/lock/aiv-gpu.lock      # shared by every WORK dir on this machine (one GPU)
