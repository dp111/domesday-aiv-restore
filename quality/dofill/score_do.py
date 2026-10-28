#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Score dropout fills inside the flagged pixels against the truth (see prep.py).
  score_do.py   -> table per segment: RMS error (8-bit) of Y and of Cb/Cr inside the dropout mask
Reference fix4_lv = a run_bench.sh label decoding the same stack WITHOUT the added dropouts (the floor). Variants: dfA..dfD (newdecode outputs), E = dfA + decoded-picture interpolation from the nearest clean rows
above/below in the same field (rows +-2, +-4 ...), E1 = same with the other field's rows +-1 allowed; E1C = E1 applied to dfC (production), mixC = mean of dfC and E1C.
"""
import numpy as np, os, sys
from scipy.ndimage import binary_dilation
B = os.environ.get('FQ_BENCH') or sys.exit('set FQ_BENCH=<make_bench.py benchmark dir>'); FW, FH = 928, 576
SEGS = ['seg03000', 'seg08000', 'seg16256', 'seg29696', 'seg50112', 'pat40000']
def interp(o, M, step_rows):
    """fill masked pixels of o (3,H,W) by linear interpolation between nearest unmasked rows (offsets in step_rows)"""
    out = o.copy(); H = o.shape[1]
    for s in (-1, 1):
        pass
    va = np.full(o.shape, np.nan); da = np.full(M.shape, np.inf); vb = va.copy(); db = da.copy()
    for d in sorted(step_rows, key=abs):
        sh = np.full(o.shape, np.nan); ok = np.zeros(M.shape, bool)
        r0, r1 = max(0, -d), min(H, H - d)
        sh[:, r0:r1] = o[:, r0 + d:r1 + d]; ok[r0:r1] = ~M[r0 + d:r1 + d]
        if d < 0:
            b = ok & M & ~np.isfinite(da); va[:, b] = sh[:, b]; da[b] = -d
        else:
            b = ok & M & ~np.isfinite(db); vb[:, b] = sh[:, b]; db[b] = d
    both = np.isfinite(da) & np.isfinite(db)
    w = np.where(both, db / np.where(both, da + db, 1), np.where(np.isfinite(da), 1.0, 0.0))
    f = np.where(np.isfinite(va), va, 0) * w + np.where(np.isfinite(vb), vb, 0) * (1 - w)
    sel = M & (np.isfinite(da) | np.isfinite(db))
    out[:, sel] = f[:, sel]
    return out
labs = [l for l in ('fix4_lv', 'dfA', 'dfB', 'dfC', 'dfD', 'dfCr', 'dfCm', 'dfCt') if all(os.path.exists(f'{B}/{s}/{l}/out.yuv') for s in SEGS)]
cols = labs + (['E', 'E1'] if 'dfA' in labs else []) + (['E1C', 'mixC'] if 'dfC' in labs else [])
tot = {c: [0.0, 0.0, 0] for c in cols}
print('RMS error inside dropouts, 8-bit levels, Y / C   (' + '  '.join(cols) + ')')
for s in SEGS:
    T = np.memmap(f'{B}/{s}/truth.yuv', '<u2', 'r').reshape(-1, 3, FH, FW)
    M = np.load(f'{B}/{s}/dofill/mask.npy')
    O = {l: np.memmap(f'{B}/{s}/{l}/out.yuv', '<u2', 'r').reshape(-1, 3, FH, FW) for l in labs}
    n = min(len(T), len(M), *(len(v) for v in O.values()))
    acc = {c: [0.0, 0.0, 0] for c in cols}
    for k in range(n):
        m = binary_dilation(M[k], np.ones((1, 5), bool)); m[:8] = m[-8:] = False; m[:, :16] = m[:, -16:] = False
        if not m.any(): continue
        t = T[k].astype(np.float64)
        outs = {l: O[l][k].astype(np.float64) for l in labs}
        if 'dfA' in labs:
            outs['E'] = interp(outs['dfA'], m, [-8, -6, -4, -2, 2, 4, 6, 8])
            outs['E1'] = interp(outs['dfA'], m, [-4, -3, -2, -1, 1, 2, 3, 4])
        if 'dfC' in labs:
            outs['E1C'] = interp(outs['dfC'], m, [-4, -3, -2, -1, 1, 2, 3, 4])
            outs['mixC'] = 0.5 * (outs['dfC'] + outs['E1C'])
        for c in cols:
            e = outs[c] - t
            acc[c][0] += (e[0][m] ** 2).sum(); acc[c][1] += (e[1:][:, m] ** 2).sum() / 2; acc[c][2] += m.sum()
    line = []
    for c in cols:
        a = acc[c]; [tot[c].__setitem__(i, tot[c][i] + a[i]) for i in range(3)]
        line.append(f'{np.sqrt(a[0] / a[2]) / 256:5.2f}/{np.sqrt(a[1] / a[2]) / 256:4.2f}')
    print(f'{s:9s} ' + '  '.join(line), flush=True)
print(f'{"all":9s} ' + '  '.join(f'{np.sqrt(tot[c][0] / tot[c][2]) / 256:5.2f}/{np.sqrt(tot[c][1] / tot[c][2]) / 256:4.2f}' for c in cols))
