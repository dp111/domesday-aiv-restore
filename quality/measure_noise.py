"""Measure the real disc + player noise model of a side from per-capture TBC segments (fq benchmark).

Uses the blank vertical-interval lines (field rows 5-14 and 20-21: flat black, no VITS / VBI data), on
fields matched across captures by CAV picture number, so the noise of different captures of one picture
can be compared sample by sample:

  PSD_i(f)            noise power spectrum of capture i
  shared_m(f)         cross-spectrum Re<X_i X_j*> averaged over pairs of captures of master m: the noise
                      pressed into the master (identical in every pressing / capture of it)
  indep_i(f)          PSD_i - shared_m: the capture's own (player / RF) noise
  cross(f)            cross-spectrum between captures of DIFFERENT masters (should be ~0)

Writes an .npz noise model (f, shared per master, indep per capture) used by make_bench.py.
  measure_noise.py <out.npz> <masters 'ds1,ds2;ds5,ds11;ds3'> <segdir>...
"""
import sys, glob, os, json, itertools
import numpy as np

sys.path.insert(0, os.environ.get('LD', os.path.expanduser('~/ld-decode')))
from lddecode.stack import cav_picture, field_vbi

W, H, FS = 1135, 313, 17734475.0
ROWS = list(range(5, 15)) + [20, 21]
X0, X1 = 200, 1050
N = X1 - X0
WIN = np.hanning(N); WN = (WIN ** 2).sum()


def load(seg, cap):
    base = glob.glob(f'{seg}/*_{cap}.tbc')[0]
    x = np.memmap(base, dtype='<u2', mode='r').reshape(-1, H, W)
    js = json.load(open(base + '.json'))
    pics = {}
    for k, fj in enumerate(js['fields'][:len(x)]):
        p = cav_picture(field_vbi(fj))
        if p is not None:
            pics[(p, bool(fj.get('isFirstField', k % 2 == 0)))] = k
    return x, pics


def spec(rows):
    r = rows.astype(np.float64)
    r = r - r.mean(1, keepdims=True)
    return np.fft.rfft(r * WIN, axis=1)


def main():
    out, mspec, segs = sys.argv[1], sys.argv[2], sys.argv[3:]
    masters = [g.split(',') for g in mspec.split(';')]
    caps = [c for g in masters for c in g]
    mof = {c: i for i, g in enumerate(masters) for c in g}
    nf = N // 2 + 1
    P = {c: np.zeros(nf) for c in caps}; nP = {c: 0 for c in caps}
    C = {}; nC = {}
    for seg in segs:
        data = {c: load(seg, c) for c in caps}
        common = set.intersection(*[set(d[1]) for d in data.values()])
        for key in sorted(common):
            S = {}
            for c in caps:
                x, pics = data[c]
                f = x[pics[key]][ROWS, X0:X1]
                if abs(np.median(f) - 16000) > 800 or f.std() > 3000:   # not a blank black line set
                    break
                S[c] = spec(f)
            else:
                for c in caps:
                    P[c] += (np.abs(S[c]) ** 2).mean(0); nP[c] += 1
                for a, b in itertools.combinations(caps, 2):
                    k = (a, b)
                    C[k] = C.get(k, 0) + (S[a] * np.conj(S[b])).real.mean(0); nC[k] = nC.get(k, 0) + 1
    f = np.fft.rfftfreq(N, 1 / FS)
    norm = 2.0 / (WN * FS)      # one-sided PSD, units^2 / Hz
    P = {c: P[c] / max(nP[c], 1) * norm for c in caps}
    C = {k: v / nC[k] * norm for k, v in C.items()}
    shared = []
    for m, g in enumerate(masters):
        pr = [C[k] for k in C if mof[k[0]] == m and mof[k[1]] == m]
        shared.append(np.mean(pr, 0) if pr else np.zeros(nf))
    cross = np.mean([C[k] for k in C if mof[k[0]] != mof[k[1]]], 0)
    indep = {c: np.maximum(P[c] - shared[mof[c]], 0) for c in caps}
    bands = [(0, 2e6), (2e6, 3.5e6), (3.5e6, 5.5e6), (5.5e6, FS / 2)]
    def bp(p): return [np.sum(p[(f >= lo) & (f < hi)]) * (f[1] - f[0]) for lo, hi in bands]
    print(f'fields used per capture: {nP[caps[0]]}  (segments {len(segs)})')
    print('noise power per band (16-bit units^2) ' + ' '.join(f'{lo/1e6:.1f}-{hi/1e6:.1f}MHz' for lo, hi in bands))
    for c in caps:
        print(f'  {c:5s} m{mof[c]} total ' + ' '.join(f'{v:9.0f}' for v in bp(P[c])) + '   indep ' + ' '.join(f'{v:9.0f}' for v in bp(indep[c])))
    for m in range(len(masters)):
        print(f'  shared master {m}:      ' + ' '.join(f'{v:9.0f}' for v in bp(shared[m])))
    print(f'  cross-master (should be ~0): ' + ' '.join(f'{v:9.0f}' for v in bp(cross)))
    np.savez(out, f=f, fs=FS, n=N, masters=np.array(mspec), caps=np.array(caps),
             shared=np.array(shared), indep=np.array([indep[c] for c in caps]), total=np.array([P[c] for c in caps]),
             cross=cross)
    print('wrote', out)


if __name__ == '__main__':
    main()
