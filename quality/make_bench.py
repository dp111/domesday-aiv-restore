"""Build a synthetic multi-capture benchmark with a known true picture (fq benchmark, 2026-10-24).

For each segment of real decoded frames (the "truth", any content will do: it is the reference by
definition), encode a clean PAL composite TBC with ld-chroma-encoder, then make one synthetic capture per
real capture of the side:

  capture c of master m = subsample_shift(H(clean) + shared_m, dx_c) + indep_c, + real dropouts

  H          the disc channel's slow edge smear (paldec deghost.py model, c=0.16 tau=9 samples, FQ_SMEAR=c,tau;
             0 = off): every real TBC carries it and newdecode always applies its inverse G, so a bench
             without it is too sharp (2026-10-26: luma gain 1-2 MHz 1.16 on the noise-free decode = 1/|H|).
             clean.tbc holds H(clean), the signal the stack should recover.  The noise is added after H (the
             measured spectra are of real TBCs, i.e. already through the channel).

  shared_m   one Gaussian noise realisation per master and segment, with the master's measured spectrum
             (noise pressed into the glass master: identical in every capture of it)
  indep_c    the capture's own measured player / RF noise spectrum, independent per capture
  dx_c       a fixed sub-sample timing offset per master (FQ_MASTER_DX, default 0,-0.3,-1.3 as measured on
             South) + +-0.02 per capture, so the stacker's cross-master registration is exercised
  chroma     per-capture per-line chroma phase / gain errors as measured between real captures (FQ_CHROMA)
  dropouts   real per-field dropOuts lists of the corresponding real capture (cycled), written into
             the json and the span replaced by dropout-like signal

The json carries the real CAV picture numbers (VBI) of the truth frames, so the stacker aligns the
synthetic captures exactly as it does real ones.  Noise is level-independent in this model (blank-line
spectra); the model is in measure_noise.py.

  make_bench.py <benchdir> <noise.npz> <truth.mkv> <stack.tbc.json> <dropout-segdir-glob> <start>:<len>...

FQ_FROM=<old benchdir>: rebuild an existing benchmark (same segment list and order as it was made): truth.yuv,
VBI and every capture's dropOuts are taken from it, so with the same seed the noise realisations are
identical and only the changed options (e.g. FQ_SMEAR) differ.  <truth.mkv> <stack.tbc.json> <glob> are then
ignored ('-').
"""
import sys, os, glob, json, subprocess
import numpy as np
from scipy.signal import lfilter

W, H = 1135, 313
FW, FH = 928, 576


def psd_line_filter(f_model, psd, n, fs):
    """Amplitude response for an n-sample line so irfft(rfft(randn) * h) has the PSD psd."""
    fk = np.fft.rfftfreq(n, 1 / fs)
    p = np.interp(fk, f_model, psd, left=psd[0], right=psd[-1])
    return np.sqrt(np.maximum(p, 0) * fs / 2)


def noise_fields(rng, h, nfields):
    g = np.fft.rfft(rng.standard_normal((nfields, H, W)), axis=2)
    return np.fft.irfft(g * h, n=W, axis=2)


def shift_lines(x, dx):
    if dx == 0:
        return x
    k = np.fft.rfftfreq(W)
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * np.exp(-2j * np.pi * k * dx), n=W, axis=-1)


# per-capture chroma (player pilot / timebase) errors, measured between real South captures: per-line phase
# difference 1.5-5 deg rms (Test Card 4.95, p95 8.8, max 15.5), amplitude ratio sd 0.025-0.034.  Model: per
# field a smooth phase drift (FQ_CHROMA amplitude deg) + per-line jitter (deg) + per-line chroma gain sd.
CHROMA = [float(v) for v in os.environ.get('FQ_CHROMA', '3.0,2.4,0.03').split(',')]
_FK = np.fft.rfftfreq(W, 1 / 17734475.0)
_CB = (_FK > 3.3e6) & (_FK < 5.6e6)


def chroma_errors(rng, x):
    """Rotate / scale the chroma band of every line of x (fields, H, W) by a capture's own errors."""
    if not any(CHROMA):
        return x
    nf = x.shape[0]
    t = np.linspace(0, 1, H)[None, :]
    ph = (np.radians(CHROMA[0]) * np.sin(2 * np.pi * (t * rng.uniform(0.3, 1.5, (nf, 1)) + rng.uniform(0, 1, (nf, 1))))
          + np.radians(CHROMA[1]) * rng.standard_normal((nf, H)))
    g = 1 + CHROMA[2] * rng.standard_normal((nf, H))
    X = np.fft.rfft(x, axis=-1)
    X[..., _CB] *= (g * np.exp(1j * ph))[..., None]
    return np.fft.irfft(X, n=W, axis=-1)


SMEAR = [float(v) for v in os.environ.get('FQ_SMEAR', '0.16,9').split(',')]


def smear(fields):
    """the channel H(z) = ((1-ca) - (1-c)a z^-1) / (1 - a z^-1), exact inverse of deghost.deghost's G, run per
    field over its lines as one signal (as deghost is applied in newdecode)"""
    if not SMEAR[0]:
        return fields
    c, tau = SMEAR; a = np.exp(-1.0 / tau)
    out = np.empty_like(fields)
    for k, f in enumerate(fields):
        fl = f.reshape(-1)
        out[k] = (lfilter([1.0 - c * a, -(1.0 - c) * a], [1.0, -a], fl - fl[0]) + fl[0]).reshape(f.shape)
    return out


def dropout_pool(segglob, cap):
    pool = []
    for seg in sorted(glob.glob(segglob)):
        for j in glob.glob(f'{seg}/*_{cap}.tbc.json'):
            for fj in json.load(open(j))['fields']:
                do = fj.get('dropOuts') or {}
                pool.append(list(zip(do.get('fieldLine', []), do.get('startx', []), do.get('endx', []))))
    return pool or [[]]


def main():
    bench, npz, mkv, sjson, segglob = sys.argv[1:6]
    old = os.environ.get('FQ_FROM')
    # <start>:<len> takes frames start.. of truth.mkv; file=<yuv>:<start>:<len> takes a ready truth .yuv
    # (928x576 yuv444p16le, e.g. test patterns from make_patterns.sh) and only borrows the VBI of frames start..
    segs = []
    for a in sys.argv[6:]:
        if a.startswith('file='):
            pth, st, ln = a[5:].rsplit(':', 2); segs.append((int(st), int(ln), pth))
        else:
            st, ln = map(int, a.split(':')); segs.append((st, ln, None))
    M = np.load(npz)
    f, fs = M['f'], float(M['fs'])
    masters = [g.split(',') for g in str(M['masters']).split(';')]
    caps = list(M['caps'])
    shared = list(M['shared'])
    indep = {c: M['indep'][i] for i, c in enumerate(caps)}
    total = {c: M['total'][i] for i, c in enumerate(caps)}
    measured = [m for m, g in enumerate(masters) if len(g) > 1]
    for m, g in enumerate(masters):          # single-capture master: its shared part cannot be measured;
        if len(g) == 1:                      # assume the others' shared spectrum, the rest independent
            shared[m] = np.mean([shared[k] for k in measured], 0)
            indep[g[0]] = np.maximum(total[g[0]] - shared[m], 0)
    hs = [psd_line_filter(f, s, W, fs) for s in shared]
    hi = {c: psd_line_filter(f, indep[c], W, fs) for c in caps}
    rng = np.random.default_rng(20261024)
    # timing offsets: captures of one master lock to the same pressing's sync and agree to +-0.03 samples
    # (measured, South impulse_test segments); the masters differ: South ds5/ds11 -0.22..-0.43, ds3 -1.3
    mdx = [float(v) for v in os.environ.get('FQ_MASTER_DX', '0,-0.3,-1.3').split(',')]
    dx = {c: (mdx[m] if m < len(mdx) else 0.0) + (0.0 if c == masters[0][0] else float(rng.uniform(-0.02, 0.02)))
          for m, g in enumerate(masters) for c in g}
    pools = {c: dropout_pool(segglob, c) for c in caps} if not old else None
    pos = {c: 0 for c in caps}
    SJ = json.load(open(sjson)) if not old else None
    os.makedirs(bench, exist_ok=True)
    meta = {'noise_model': npz, 'truth': mkv, 'masters': str(M['masters']), 'dx': dx, 'smear': SMEAR,
            'from': old, 'segments': []}
    if os.path.exists(f'{bench}/bench.json'):          # adding segments to an existing benchmark
        old = json.load(open(f'{bench}/bench.json'))
        meta['segments'] = [g for g in old['segments'] if g['dir'] not in
                            {f'{bench}/{"pat" if src else "seg"}{st:05d}' for st, _, src in segs}]
    for s, L, src in segs:
        d = f'{bench}/{"pat" if src else "seg"}{s:05d}'
        os.makedirs(d, exist_ok=True)
        tr = f'{d}/truth.yuv'
        od = f'{old}/{os.path.basename(d)}' if old else None
        if old:
            subprocess.run(['cp', f'{od}/truth.yuv', tr], check=True)
        elif src:
            with open(src, 'rb') as fi, open(tr, 'wb') as fo:
                fo.write(fi.read(L * 3 * FW * FH * 2))
        else:
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{s / 25:.3f}', '-i', mkv, '-frames:v', str(L),
                            '-an', '-f', 'rawvideo', '-pix_fmt', 'yuv444p16le', tr], check=True)
        subprocess.run(['ld-chroma-encoder', '-q', '-f', 'PAL', '-p', 'yuv', tr, f'{d}/clean.tbc'], check=True)
        ej = json.load(open(f'{d}/clean.tbc.json'))
        nf = len(ej['fields'])
        oj = json.load(open(f'{od}/clean.tbc.json')) if old else None
        for k in range(nf):                   # real VBI (CAV picture numbers) of the truth frames
            fsrc = oj['fields'][k] if old else SJ['fields'][2 * s + k]
            ej['fields'][k]['vbi'] = fsrc.get('vbi', {})
            ej['fields'][k]['seqNo'] = k + 1
        json.dump(ej, open(f'{d}/clean.tbc.json', 'w'))
        clean = smear(np.fromfile(f'{d}/clean.tbc', dtype='<u2').reshape(nf, H, W).astype(np.float64)).astype(np.float32)
        np.clip(np.rint(clean), 0, 65535).astype('<u2').tofile(f'{d}/clean.tbc')
        cjs = {c: json.loads(json.dumps(ej)) for c in caps}
        if old:                               # the old bench's per-field dropout lists, in its order
            odl = {c: [(fj.get('dropOuts') or {}) for fj in json.load(open(f'{od}/syn_{c}.tbc.json'))['fields']]
                   for c in caps}
        outs = {c: open(f'{d}/syn_{c}.tbc', 'wb') for c in caps}
        B = 16                                # fields per block (memory: ~1 GB peak)
        for k0 in range(0, nf, B):
            k1 = min(nf, k0 + B)
            cb = clean[k0:k1].astype(np.float64)
            for m, g in enumerate(masters):
                disc = cb + noise_fields(rng, hs[m], k1 - k0)     # what is on this master's pressings
                for c in g:
                    x = chroma_errors(rng, shift_lines(disc, dx[c])) + noise_fields(rng, hi[c], k1 - k0)
                    for k in range(k0, k1):
                        if old:
                            do = odl[c][k]; dl = list(zip(do.get('fieldLine', []), do.get('startx', []), do.get('endx', [])))
                        else:
                            dl = pools[c][pos[c] % len(pools[c])]; pos[c] += 1
                        fl, sx, ex = [], [], []
                        for line, a, b in dl:
                            line, a, b = int(line), int(a), int(b)
                            # fieldLine is 1-based (ld-decode writes line+1; ld-dropout-correct and newdecode read
                            # row fieldLine-1).  Before 2026-10-26 the span was written one row too low.
                            if not (0 < line <= H and 0 <= a < b <= W):
                                continue
                            x[k - k0, line - 1, a:b] = 16384 + 3000 * rng.standard_normal(b - a)
                            fl.append(line); sx.append(a); ex.append(b)
                        if fl:
                            cjs[c]['fields'][k]['dropOuts'] = {'fieldLine': fl, 'startx': sx, 'endx': ex}
                    np.clip(np.rint(x), 0, 65535).astype('<u2').tofile(outs[c])
        for c in caps:
            outs[c].close(); json.dump(cjs[c], open(f'{d}/syn_{c}.tbc.json', 'w'))
        meta['segments'].append({'start': s, 'frames': L, 'dir': d, 'truth_src': (od or src or mkv)})
        print(f'segment {s}+{L}: {nf} fields, {len(caps)} synthetic captures', flush=True)
    json.dump(meta, open(f'{bench}/bench.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
