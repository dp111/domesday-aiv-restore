# SPDX-License-Identifier: GPL-3.0-or-later
"""Rotation-coherent drop-out refinement for PAL CAV captures (cavdo, 2026-10-27; prototype, NOT adopted).

  cavdo.py <cap.tbc> [-o out.tbc.db | --inplace] [--json] [--jobs N] [--mode click|coh|line|none]

Adds runs to a capture's ld-decode drop-out list from the capture's own TBC (the .tbc is not touched):
  z      FM-click score: rms envelope (5 samples) of the > 5.6 MHz residual, over its per-field median
         (mode line = coh with a per (line, 64-sample bin) median over +-SIGW same-parity tracks, a threshold
         learned over revolutions; SC=i: min |residual| of 4 chroma-cancelling predictors instead)
  click  add z > T_HI
  coh    also add z > T_LO where z > T_LO (or an ld-decode flag) recurs at the same line, x+-2 in >= M of
         the +-K neighbouring tracks: one CAV revolution is one frame, so a disc defect 10-100 um across
         hits the same (parity, line, x) in adjacent frames (same parity field 2, 4 .. 2K away; VBI
         picture numbers must be within K, so player jumps break the chain)
  out    ld-decode flags + added runs (widened by PAD), as drop_outs (field_id 0-based, field_line 1-based,
         endx exclusive) of a copy of the .tbc.db; --json also writes the .tbc.json (db2json)
Result on CommunitySouth 3000/8000 vs a same-master median truth: click raises the flagged share of
damage (16->28%, 8->15%) at ~40-50% precision of the added runs; coh and line add nothing over click at
the same flag budget (content repeats across tracks in held stills / slow video, so coherent support is
mostly static picture detail); downstream gain ~0 (stack -2% residual defects, ld-dropout-correct worse).
"""
import argparse, os, shutil, sqlite3, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter1d, uniform_filter1d, convolve1d
from scipy.signal import firwin

W, H = 1135, 313
FB = W * H
AX0, AX1 = 185, 1107            # active video (videoParameters activeVideoStart/End)
L0, L1 = 22, 310                # field rows searched (0-based)
P = dict(K=3, M=1, T_HI=3.5, T_LO=2.5, PAD=2, SIGW=8, XB=64)


def iscore(fld):
    x = fld.astype(np.float32)
    p = np.pad(x, ((2, 2), (4, 4)), mode='edge')
    c = p[2:-2, 4:-4]
    s = np.abs(c - 0.5 * (p[2:-2, :-8] + p[2:-2, 8:]))
    for a, b in ((p[:-4, 4:-4], p[4:, 4:-4]), (p[:-4, :-8], p[4:, 8:]), (p[:-4, 8:], p[4:, :-8])):
        np.minimum(s, np.abs(c - 0.5 * (a + b)), out=s)
    return s


_HP = -firwin(21, 5.6e6, fs=17734475.0); _HP[10] += 1           # > 5.6 MHz: above the PAL video band


def hscore(fld):
    """FM click score: rms envelope (5 samples) of the out-of-band (> 5.6 MHz) residual; a demodulator click
    rings there at 3000-6000 (16-bit) while picture content has almost none."""
    h = convolve1d(fld.astype(np.float32), _HP.astype(np.float32), axis=1, mode='nearest')
    return np.sqrt(uniform_filter1d(h * h, 5, axis=1))


def binmed(s, xb):
    """per (line, x-bin) median of the score (active area), broadcast back to H x W"""
    a = s[:, AX0:AX0 + (AX1 - AX0) // xb * xb].reshape(H, -1, xb)
    m = np.median(a, 2)
    return m


def expand(m, xb):
    o = np.repeat(m, xb, 1)
    out = np.empty((H, W), np.float32)
    out[:, :AX0] = o[:, :1]; out[:, AX1:] = o[:, -1:]
    n = min(o.shape[1], AX1 - AX0)
    out[:, AX0:AX0 + n] = o[:, :n]; out[:, AX0 + n:AX1] = o[:, n - 1:n]
    return out


def flagmask(runs):
    m = np.zeros((H, W), bool)
    for l, s, e in runs:
        if 1 <= l <= H:
            m[l - 1, max(s, 0):min(e, W)] = True
    return m


def runs_of(mask):
    out = []
    for l in np.nonzero(mask.any(1))[0]:
        r = np.diff(np.concatenate(([0], mask[l].view(np.int8), [0])))
        for s, e in zip(np.nonzero(r == 1)[0], np.nonzero(r == -1)[0]):
            out.append((int(l) + 1, int(s), int(e)))
    return out


def detect(tbc, flags, pics, f0, f1, mode='click', p=P):
    """New drop-out run lists for fields f0..f1-1.  tbc: (nfields, H, W) array; flags: list of run lists;
    pics: per-field CAV picture number or None."""
    K, sw, xb = p['K'], p['SIGW'], p['XB']
    nf = len(tbc)
    far = 2 * p.get('FAR', 0)            # evaluation control: neighbours FAR frames away (not adjacent tracks)
    span = 2 * max(K, sw if mode == 'line' else 0)
    a0, a1 = (0, nf // 2 * 2) if far else (max(0, f0 - span), min(nf, f1 + span))
    S, B = {}, {}
    def sc(f):
        if f not in S:
            S[f] = hscore(tbc[f]) if p.get('SC', 'h') == 'h' else iscore(tbc[f])
            B[f] = binmed(S[f], xb)
        return S[f]
    def bm(f):
        sc(f); return B[f]
    def scale(f):
        if mode == 'line':
            nb = [g for g in range(f - 2 * sw, f + 2 * sw + 1, 2) if a0 <= g < a1 and near(f, g, sw)]
            m = np.median(np.stack([bm(g) for g in nb]), 0)
        else:
            m = np.full_like(bm(f), np.median(bm(f)[L0:L1]))
        return expand(np.maximum(m, 1.0) / 0.6745, xb)
    def near(f, g, k):
        a, b = pics[f], pics[g]
        return a is None or b is None or abs(a - b) <= k
    Z, LO = {}, {}
    out = []
    zf = lambda f: Z[f] if f in Z else Z.setdefault(f, sc(f) / scale(f))
    st = np.ones((1, 2 * p['PAD'] + 1), bool)
    act = np.zeros((H, W), bool); act[L0:L1, AX0:AX1] = True
    for f in range(f0, f1):
        F = flagmask(flags[f])
        if mode == 'none':
            out.append(runs_of(F)); continue
        z = zf(f)
        add = (z > p['T_HI']) & act
        if mode != 'click':
            lo = (z > p['T_LO']) & act
            sup = np.zeros((H, W), np.int8); fl = np.zeros((H, W), bool)
            for d in range(1, K + 1):
                for g in ((f - 2 * d - far) % a1, (f + 2 * d + far) % a1) if far else (f - 2 * d, f + 2 * d):
                    if a0 <= g < a1 and (far or near(f, g, K)):
                        if g not in LO:
                            LO[g] = maximum_filter1d(((zf(g) > p['T_LO']) & act).view(np.uint8), 5, axis=1).astype(bool)
                        sup += LO[g]
                        fl |= flagmask(flags[g])
            add |= lo & ((sup >= p['M']) | binary_dilation(fl, np.ones((1, 5), bool)))
        add = binary_dilation(add, st)
        out.append(runs_of(F | add))
        if not far:
            for g in [g for g in S if g < f - 2 * K - 2 * sw - 2]:
                S.pop(g); B.pop(g); Z.pop(g, None); LO.pop(g, None)
    return out


# ---------------------------------------------------------------- capture I/O
def read_meta(db):
    c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    nf = c.execute('SELECT max(field_id) FROM field_record').fetchone()[0] + 1
    flags = [[] for _ in range(nf)]
    for fid, l, s, e in c.execute('SELECT field_id, field_line, startx, endx FROM drop_outs'):
        if fid < nf:
            flags[fid].append((l, s, e))
    pics = [None] * nf
    for fid, *v in c.execute('SELECT field_id, vbi0, vbi1, vbi2 FROM vbi'):
        for w in v:
            h = '%06x' % w
            if h[0] == 'f' and h[1:].isdigit():          # CAV picture number (as lddecode.stack.cav_picture)
                pics[fid] = int(h[1:]); break
    for f in range(0, nf - 1, 2):                         # picture number on one field of the frame
        if pics[f] is None: pics[f] = pics[f + 1]
        elif pics[f + 1] is None: pics[f + 1] = pics[f]
    c.close()
    return flags, pics


_G = {}
def _init(g):
    _G.update(g)


def _work(args):
    f0, f1 = args
    tbc = np.memmap(_G['tbc'], np.uint16, 'r')
    tbc = tbc[:tbc.size // FB * FB].reshape(-1, H, W)
    return f0, detect(tbc, _G['flags'], _G['pics'], f0, f1, _G['mode'], _G['p'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tbc')
    ap.add_argument('-o', '--out', help='output .tbc.db (default <tbc>.cavdo.db)')
    ap.add_argument('--inplace', action='store_true', help='replace the capture .tbc.db (atomic; keeps <db>.precavdo)')
    ap.add_argument('--json', action='store_true', help='also write the matching .tbc.json (db2json)')
    ap.add_argument('--mode', default='click', choices=['click', 'coh', 'line', 'none'])
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=400, help='fields per work unit')
    for k, v in P.items():
        ap.add_argument('--' + k.lower(), type=type(v), default=v)
    a = ap.parse_args()
    p = {k: getattr(a, k.lower()) for k in P}
    db = a.tbc + '.db'
    flags, pics = read_meta(db)
    nf = min(len(flags), os.path.getsize(a.tbc) // (FB * 2))
    _G.update(tbc=a.tbc, flags=flags, pics=pics, mode=a.mode, p=p)
    t0 = time.time()
    new = [None] * nf
    jobs = [(f, min(f + a.chunk, nf)) for f in range(0, nf, a.chunk)]
    with Pool(a.jobs, initializer=_init, initargs=(_G,)) as pool:
        for f0, r in pool.imap_unordered(_work, jobs):
            new[f0:f0 + len(r)] = r
    n0 = sum(len(x) for x in flags[:nf]); n1 = sum(len(x) for x in new)
    out = a.out or (a.tbc + '.cavdo.db')
    if a.inplace:
        out = db
    tmp = out + '.tmp'
    shutil.copyfile(db, tmp)
    c = sqlite3.connect(tmp)
    cid = c.execute('SELECT capture_id FROM capture LIMIT 1').fetchone()[0]
    c.execute('DELETE FROM drop_outs')
    c.executemany('INSERT OR IGNORE INTO drop_outs (capture_id, field_id, field_line, startx, endx) VALUES (?,?,?,?,?)',
                  ((cid, f, l, s, e) for f, rs in enumerate(new) for l, s, e in rs))
    c.commit(); c.close()
    if a.inplace and not os.path.exists(db + '.precavdo'):
        shutil.copyfile(db, db + '.precavdo')
    os.replace(tmp, out)
    print(f'{a.tbc}: {nf} fields, drop-out runs {n0} -> {n1}, {time.time() - t0:.0f} s', file=sys.stderr)
    if a.json:
        sys.path.insert(0, os.environ.get('LD', '/mnt/s/domsday/ld-decode') + '/scripts')
        import db2json
        if out.endswith('.tbc.db'):
            db2json.convert(out)
        else:
            print('--json: output name must end in .tbc.db', file=sys.stderr)


if __name__ == '__main__':
    main()
