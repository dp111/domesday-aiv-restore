#!/usr/bin/env python3
"""Dropout-fill test (fq, 2026-10-24; user: "use the line before and the line afterwards as well").
  prep.py <segdir>   -> <segdir>/dofill/{inj.tbc,inj.tbc.json, avg.tbc,avg.tbc.json, mask.npy}
inj  = the fix4 stacked composite with syn_ds3's real dropout shapes added (value 16384 + 3000 N(0,1), as
       make_bench does), flagged in the json together with the stack's own leftovers.
avg  = inj with every flagged sample replaced by linear interpolation between the nearest phase-matched
       clean lines above and below, from either field (PAL: same field +-4 lines; other field N-3 / N+1
       for a first-field line, N-1 / N+3 for a second-field line - the lines ld-dropout-correct copies),
       weighted by frame-line distance.  Searched up to 17 frame lines each way; one side only -> copy.
mask.npy = flagged samples in picture coordinates (frames, 576, 928), for scoring.
"""
import sys, json, numpy as np
W, H, X0, LINE0, OW, OH = 1135, 313, 182, 22, 928, 576
d = sys.argv[1]; o = d + '/dofill'
import os; os.makedirs(o, exist_ok=True)
src = np.memmap(d + '/fix4/stack.tbc', '<u2', 'r').reshape(-1, H, W)
js = json.load(open(d + '/fix4/stack.tbc.json')); pj = json.load(open(d + '/syn_ds3.tbc.json'))
nf = len(src) // 2 * 2
rng = np.random.default_rng(1)
x = np.array(src[:nf], np.float64); F = np.zeros((nf, H, W), bool)
for k in range(nf):
    for fj in (js['fields'][k], pj['fields'][k]):
        dd = fj.get('dropOuts') or {}
        for L, a, b in zip(dd.get('fieldLine', []), dd.get('startx', []), dd.get('endx', [])):
            if 0 < L <= H and 0 <= a < b <= W: F[k, L - 1, a:b] = True
    m = np.zeros((H, W), bool)
    dd = pj['fields'][k].get('dropOuts') or {}
    for L, a, b in zip(dd.get('fieldLine', []), dd.get('startx', []), dd.get('endx', [])):
        if 0 < L <= H and 0 <= a < b <= W: m[L - 1, a:b] = True
    x[k][m] = 16384 + 3000 * rng.standard_normal(m.sum())
def flags_json(j):
    for k in range(nf):
        L, sx, ex = [], [], []
        for r in np.where(F[k].any(1))[0]:
            v = np.diff(np.concatenate([[0], F[k, r].astype(np.int8), [0]]))
            for a, b in zip(np.where(v == 1)[0], np.where(v == -1)[0]):
                L.append(int(r + 1)); sx.append(int(a)); ex.append(int(b))
        j['fields'][k]['dropOuts'] = {'fieldLine': L, 'startx': sx, 'endx': ex} if L else {}
    j['fields'] = j['fields'][:nf]; j['videoParameters']['numberOfSequentialFields'] = nf
    return j
J = flags_json(json.loads(json.dumps(js)))
np.clip(np.rint(x), 0, 65535).astype('<u2').tofile(o + '/inj.tbc'); json.dump(J, open(o + '/inj.tbc.json', 'w'))
# interpolation
y = x.copy()
for i in range(nf // 2):
    for p in (0, 1):                       # p 0 first field (frame row 2l), 1 second (2l+1)
        k = 2 * i + p; q = 2 * i + 1 - p
        cands = []                         # (signed frame distance, field index, line offset)
        off = -3 if p == 0 else -1
        for j in range(-4, 5):
            cands.append((8 * j, k, 4 * j))
            m = off + 4 * j; cands.append((2 * m + (1 - p) - p, q, m))   # frame row of other field line l+m minus 2l+p
        best = {s: [np.full((H, W), np.nan), np.full((H, W), np.inf)] for s in (-1, 1)}
        for dist, fk, lo in cands:
            if dist == 0 or abs(dist) > 17: continue
            s = 1 if dist > 0 else -1
            v = np.full((H, W), np.nan); ok = np.zeros((H, W), bool)
            r0, r1 = max(0, -lo), min(H, H - lo)
            v[r0:r1] = x[fk, r0 + lo:r1 + lo]; ok[r0:r1] = ~F[fk, r0 + lo:r1 + lo]
            better = ok & (abs(dist) < best[s][1]) & F[k]
            best[s][0][better] = v[better]; best[s][1][better] = abs(dist)
        (va, da), (vb, db) = best[-1], best[1]
        both = np.isfinite(da) & np.isfinite(db)
        fill = np.where(both, (va * db + vb * da) / np.where(both, da + db, 1), np.where(np.isfinite(da), va, vb))
        sel = F[k] & np.isfinite(fill)
        y[k][sel] = fill[sel]
np.clip(np.rint(y), 0, 65535).astype('<u2').tofile(o + '/avg.tbc'); json.dump(J, open(o + '/avg.tbc.json', 'w'))
M = np.zeros((nf // 2, OH, OW), bool)
for i in range(nf // 2):
    for p in (0, 1):
        f = F[2 * i + p, LINE0:LINE0 + OH // 2, X0:X0 + OW]; M[i, p::2][:, :f.shape[1]] = f
np.save(o + '/mask.npy', M)
print(d, 'fields', nf, 'flagged samples/field %.0f' % (F.sum() / nf), 'unfilled by avg', int((F & ~np.isfinite(y)).sum()))
