#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Motion-adaptive temporal dropout fill, offline on a newdecode output (user 2026-10-25: "if that part of the
frame doesn't move we can use the previous and next frame").
  temporal.py <label> T1 [T2 ...]     (thresholds in 8-bit levels of local mean |dY| around the dropout)
For each frame, dropout pixels (mask.npy, +-2) take the mean of the previous / next frame's pixels where that
neighbour is locally static (box-averaged |dY| over clean pixels < T) and clean itself; else keep <label>'s value.
"""
import os, sys, numpy as np
from scipy.ndimage import binary_dilation, uniform_filter
B = os.environ.get('FQ_BENCH') or sys.exit('set FQ_BENCH=<make_bench.py benchmark dir>'); FW, FH = 928, 576
SEGS = ['seg03000', 'seg08000', 'seg16256', 'seg29696', 'seg50112', 'pat40000']
lab = sys.argv[1]; TS = [float(t) for t in sys.argv[2:]]
BY, BX = 13, 33
def boxmean(v, w):
    return uniform_filter(v * w, (BY, BX), mode='constant') / np.maximum(uniform_filter(w, (BY, BX), mode='constant'), 1e-6)
tot = {t: [0.0, 0.0, 0, 0] for t in ['base'] + TS}
print(f'{lab}: RMS error inside dropouts Y/C, and % of dropout pixels filled temporally')
for s in SEGS:
    T = np.memmap(f'{B}/{s}/truth.yuv', '<u2', 'r').reshape(-1, 3, FH, FW)
    O = np.memmap(f'{B}/{s}/{lab}/out.yuv', '<u2', 'r').reshape(-1, 3, FH, FW)
    M0 = np.load(f'{B}/{s}/dofill/mask.npy'); n = min(len(T), len(O), len(M0))
    M = np.array([binary_dilation(M0[k], np.ones((1, 5), bool)) for k in range(n)])
    acc = {t: [0.0, 0.0, 0, 0] for t in tot}
    for k in range(n):
        m = M[k].copy(); m[:8] = m[-8:] = False; m[:, :16] = m[:, -16:] = False
        if not m.any(): continue
        x = O[k].astype(np.float64); t = T[k].astype(np.float64)
        nb = []
        for j in (k - 1, k + 1):
            if 0 <= j < n:
                y = O[j].astype(np.float64); valid = (~M[k] & ~M[j]).astype(np.float64)
                d = boxmean(np.abs(x[0] - y[0]), valid) / 256
                nb.append((y, d, ~M[j]))
        for th in tot:
            if th == 'base':
                f = x; used = 0
            else:
                ssum = np.zeros_like(x); cnt = np.zeros(m.shape)
                for y, d, ok in nb:
                    st = (d < th) & ok & m
                    ssum[:, st] += y[:, st]; cnt[st] += 1
                f = x.copy(); sel = cnt > 0
                f[:, sel] = ssum[:, sel] / cnt[sel]; used = sel.sum()
            e = f - t
            a = acc[th]; a[0] += (e[0][m] ** 2).sum(); a[1] += (e[1:][:, m] ** 2).sum() / 2; a[2] += m.sum(); a[3] += used
    print(f'{s:9s} ' + '  '.join(f'{th}: {np.sqrt(a[0]/a[2])/256:5.2f}/{np.sqrt(a[1]/a[2])/256:4.2f} ({100*a[3]/a[2]:.0f}%)' for th, a in acc.items()), flush=True)
    for th in tot: tot[th] = [u + v for u, v in zip(tot[th], acc[th])]
print(f'{"all":9s} ' + '  '.join(f'{th}: {np.sqrt(a[0]/a[2])/256:5.2f}/{np.sqrt(a[1]/a[2])/256:4.2f} ({100*a[3]/a[2]:.0f}%)' for th, a in tot.items()))
