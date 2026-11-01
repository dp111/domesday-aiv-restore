# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-TBC luma equaliser from the disc's own multiburst VITS (v6_3, 2026-10-31; disc-position aware v6_4, 2026-11-01).

The deghost inverse G (deghost.py) lifts everything above ~1 MHz by 16-18 % over DC.  The disc's CCIR
multiburst (0.5 / 1 / 2 / 4 / 4.8 / 5.8 MHz packets between a 2-level reference bar; Domesday: field 0
line 20) measures the whole record/playback chain: on Community South the raw stack reads 0.91 / 0.92 /
0.93 / 1.00 / 0.90 / 0.51 of the reference, so after G the picture's luma is 1.00 / 1.06 / 1.08 / 1.17 /
1.04 at 0.5-4.8 MHz (other discs +4 .. +27 %); its 4.43 MHz chroma reads the same lift (transform3d 1.094
on the EBU bars).  The equaliser is 1 / that response (pchip through DC = 1 and the packets, held above
4.8 MHz), never a boost (<= 1) and >= EQ_FLOOR, applied to the decoded luma rows only (chroma / the solve
untouched).  It is measured once per TBC from frames spread over the WHOLE file, so chunked runs agree.
No multiburst found (synthetic bench, other VITS) -> no equaliser.

v6_4 (2026-11-01): the decoded HF response varies with disc radius (ld-decode's EQ / MTF servos settle
differently: Community South after deghost reads 4 MHz 1.18-1.20 over frames 0-2000, 1.06 at 12000, 1.03-1.07
over 14k-52k; EcoDisc is flat), so one whole-TBC curve left the first ~12k frames 5-15 % hot at 1-4 MHz.
measure_table() measures the multiburst per window of WIN frames (PER_WIN frames each, only the VITS line is
read), smooths the window medians along the frame index (running median of 3; windows with too few valid
multibursts take their neighbours' values, none at all -> the global median), and Table.gain(frame) linearly
interpolates the per-window eq_curve()s between window centres.  The table depends only on the TBC (never on
--start / --frames), so chunked runs agree; it is cached next to the TBC (<tbc>.lumaeq.json, keyed by size,
mtime, window parameters and deghost) and written atomically.
PALDEC_LUMA_EQ_MODE=global: v6_3's single whole-TBC curve (measure(), unchanged).
PALDEC_LUMA_EQ_WIN / _PER_WIN: window length / frames sampled per window (default 750 / 16).
PALDEC_LUMA_EQ_CACHE: 0 = no cache; a directory = cache there (<basename>.<path hash>.lumaeq.json); else a file path.
"""
import json, os, time
import numpy as np

FS = 17.734475
F0 = (0.5, 1.0, 2.0, 4.0, 4.8, 5.8)
SEG = ((425, 495), (535, 610), (650, 725), (758, 840), (862, 950), (972, 1062))
REF_HI, REF_LO = (230, 268), (290, 340)
EQ_FLOOR = float(os.environ.get('PALDEC_LUMA_EQ_FLOOR', 0.75))
MODE = os.environ.get('PALDEC_LUMA_EQ_MODE', 'window')
WIN = int(os.environ.get('PALDEC_LUMA_EQ_WIN', 750))
PER_WIN = int(os.environ.get('PALDEC_LUMA_EQ_PER_WIN', 16))
TABLE_VERSION = 'v6_4.1'


def _fit(s, f0):
    t = np.arange(len(s), dtype=np.float64)
    best = 0.0
    for f in np.linspace(f0 * 0.97, f0 * 1.03, 31):
        A = np.stack([np.ones_like(t), t, np.cos(2 * np.pi * f / FS * t), np.sin(2 * np.pi * f / FS * t)], 1)
        c = np.linalg.lstsq(A, s, rcond=None)[0]
        best = max(best, float(np.hypot(c[2], c[3])))
    return best


def _is_mb(l, black, white):
    """multiburst signature: high reference bar, low pedestal, a 0.5 MHz packet of about half the bar step"""
    hi, lo = l[REF_HI[0]:REF_HI[1]].mean(), l[REF_LO[0]:REF_LO[1]].mean()
    span = white - black
    if not (hi - lo > 0.3 * span and l[REF_HI[0]:REF_HI[1]].std() < 0.05 * span and l[REF_LO[0]:REF_LO[1]].std() < 0.05 * span):
        return False
    r = _fit(l[SEG[0][0]:SEG[0][1]], F0[0]) / ((hi - lo) / 2)
    s = l[SEG[5][0]:SEG[5][1]]; S = np.abs(np.fft.rfft((s - s.mean()) * np.hanning(len(s)), 1024))
    fpk = np.fft.rfftfreq(1024, 1 / FS)[S.argmax()]
    return 0.6 < r < 1.4 and 5.3 < fpk < 6.3


def measure(tbc, inverse, nsample=64, W=1135, H=313):
    """-> (response after `inverse` at F0 (median over frames), number of frames used, (field parity, line))
    or (None, 0, None) when the TBC carries no multiburst"""
    j = json.load(open(tbc[:-4] + '.tbc.json'))
    vp = j['videoParameters']; black, white = vp['black16bIre'], vp['white16bIre']
    nfr = os.path.getsize(tbc) // (2 * W * H * 2)
    if nfr == 0:
        return None, 0, None
    fd = os.open(tbc, os.O_RDONLY)
    try:
        def field(k):
            return np.frombuffer(os.pread(fd, W * H * 2, k * W * H * 2), '<u2').reshape(H, W).astype(np.float64)
        frames = np.unique(np.linspace(0, nfr - 1, min(nfr, nsample)).astype(int))
        loc = None
        for k in frames[:16]:
            for p in (0, 1):
                f = field(2 * k + p)
                for L in range(5, 24):
                    if _is_mb(f[L], black, white):
                        loc = (p, L); break
                if loc: break
            if loc: break
        if loc is None:
            return None, 0, None
        R = []
        for k in frames:
            f = field(2 * k + loc[0])
            if not _is_mb(f[loc[1]], black, white):
                continue
            g = inverse(f)[loc[1]]
            ref = (g[REF_HI[0]:REF_HI[1]].mean() - g[REF_LO[0]:REF_LO[1]].mean()) / 2
            R.append([_fit(g[a:b], f0) / ref for (a, b), f0 in zip(SEG, F0)])
    finally:
        os.close(fd)
    if len(R) < 3:
        return None, len(R), loc
    return np.median(np.array(R), 0), len(R), loc


def eq_curve(resp, ow, fmax=4.8):
    """rfft-bin gains (len ow // 2 + 1) for rows of ow samples at FS: 1 / response, pchip through (0, 1) and the
    packets up to fmax, held beyond; clipped to [EQ_FLOOR, 1]"""
    from scipy.interpolate import PchipInterpolator
    fr = [0.0] + [f for f in F0 if f <= fmax]
    rr = [1.0] + [r for f, r in zip(F0, resp) if f <= fmax]
    f = np.fft.rfftfreq(ow, 1 / FS)
    R = PchipInterpolator(fr, rr, extrapolate=False)(np.minimum(f, fr[-1]))
    return np.clip(1.0 / R, EQ_FLOOR, 1.0).astype(np.float32)


# ---- v6_4: per-window table ---------------------------------------------------------------------------------
def _locate(read_field, frames, black, white):
    for k in frames[:16]:
        for p in (0, 1):
            f = read_field(2 * k + p)
            for L in range(5, 24):
                if _is_mb(f[L], black, white):
                    return (p, L)
    return None


def sample(tbc, inverse, frames, loc=None, W=1135, H=313):
    """-> (responses (len(frames), 6), NaN rows where the line is no multiburst; loc).  Reads only the VITS line
    and the one before it (the deghost IIR's memory is ~10 samples, far shorter than the 230-sample lead to the
    reference bar, so the result equals the full-field inverse)."""
    j = json.load(open(tbc[:-4] + '.tbc.json'))
    vp = j['videoParameters']; black, white = vp['black16bIre'], vp['white16bIre']
    fd = os.open(tbc, os.O_RDONLY)
    try:
        if loc is None:
            loc = _locate(lambda k: np.frombuffer(os.pread(fd, W * H * 2, k * W * H * 2), '<u2').reshape(H, W)
                          .astype(np.float64), list(frames), black, white)
        R = np.full((len(frames), len(F0)), np.nan)
        if loc is None:
            return R, None
        p, L = loc
        for i, k in enumerate(frames):
            b = os.pread(fd, 2 * W * 2, (2 * int(k) + p) * W * H * 2 + (L - 1) * W * 2)
            if len(b) < 4 * W:
                continue
            x = np.frombuffer(b, '<u2').reshape(2, W).astype(np.float64)
            if not _is_mb(x[1], black, white):
                continue
            g = inverse(x)[1]
            ref = (g[REF_HI[0]:REF_HI[1]].mean() - g[REF_LO[0]:REF_LO[1]].mean()) / 2
            R[i] = [_fit(g[a:b_], f0) / ref for (a, b_), f0 in zip(SEG, F0)]
    finally:
        os.close(fd)
    return R, loc


def window_frames(nfr, win, per_win):
    """frames sampled in each window: per_win evenly spaced, centred in their sub-intervals"""
    out = []
    for w0 in range(0, nfr, win):
        n = min(win, nfr - w0)
        out.append(np.unique((w0 + (np.arange(per_win) + 0.5) * n / per_win).astype(int)))
    return out


def smooth(raw):
    """raw: (nwin, 6) window medians, NaN = no usable multiburst.  Running median of 3 along the window index
    (nan-aware, 2 at the ends), then windows still NaN take the linear interpolation of the valid ones
    (held flat beyond the first / last).  -> (nwin, 6) or None when no window is valid."""
    nw = len(raw)
    ok = ~np.isnan(raw[:, 0])
    if not ok.any():
        return None
    s = np.full_like(raw, np.nan)
    for w in range(nw):
        blk = raw[max(0, w - 1):w + 2]
        blk = blk[~np.isnan(blk[:, 0])]
        if len(blk):
            s[w] = np.median(blk, 0)
    ok = ~np.isnan(s[:, 0])
    idx = np.arange(nw)
    for c in range(raw.shape[1]):
        s[:, c] = np.interp(idx, idx[ok], s[ok, c])
    return s


def _cache_path(tbc):
    c = os.environ.get('PALDEC_LUMA_EQ_CACHE', '')
    if c == '0':
        return None
    if c == '':
        return tbc + '.lumaeq.json'
    if os.path.isdir(c):    # one directory for many TBCs (several are called doc.tbc): name by the full path too
        import hashlib
        h = hashlib.md5(os.path.abspath(tbc).encode()).hexdigest()[:8]
        return os.path.join(c, '%s.%s.lumaeq.json' % (os.path.basename(tbc), h))
    return c


def measure_table(tbc, inverse, win=None, per_win=None, key_extra=None, W=1135, H=313, log=None):
    """-> dict(win, per_win, nfr, loc, centres, raw (window medians, None = missing), nvalid, resp (smoothed),
    global (median of all sampled frames), cached) or None (no multiburst).  Same for every chunk of one TBC."""
    win = win or WIN; per_win = per_win or PER_WIN
    st = os.stat(tbc)
    key = dict(version=TABLE_VERSION, size=st.st_size, mtime_ns=st.st_mtime_ns, win=win, per_win=per_win,
               extra=key_extra)
    cp = _cache_path(tbc)
    if cp and os.path.exists(cp):
        try:
            d = json.load(open(cp))
            if d.get('key') == key:
                d['cached'] = cp
                return d if d.get('resp') is not None else None
        except (OSError, ValueError):
            pass
    t0 = time.time()
    nfr = st.st_size // (2 * W * H * 2)
    groups = window_frames(nfr, win, per_win)
    allf = np.concatenate(groups) if groups else np.zeros(0, int)
    # locate on frames spread over the whole TBC (as measure() does), then sample just the VITS line
    spread = np.unique(np.linspace(0, nfr - 1, min(nfr, 64)).astype(int)) if nfr else np.zeros(0, int)
    j = json.load(open(tbc[:-4] + '.tbc.json')); vp = j['videoParameters']
    fd = os.open(tbc, os.O_RDONLY)
    try:
        loc = _locate(lambda k: np.frombuffer(os.pread(fd, W * H * 2, k * W * H * 2), '<u2').reshape(H, W)
                      .astype(np.float64), list(spread), vp['black16bIre'], vp['white16bIre']) if nfr else None
    finally:
        os.close(fd)
    d = dict(key=key, win=win, per_win=per_win, nfr=nfr, loc=loc, centres=None, raw=None, nvalid=None, resp=None,
             glob=None, secs=None)
    if loc is not None:
        R, _ = sample(tbc, inverse, allf, loc=loc, W=W, H=H)
        raw, nv, i = [], [], 0
        need = max(2, (per_win + 1) // 2)
        for g in groups:
            r = R[i:i + len(g)]; i += len(g)
            r = r[~np.isnan(r[:, 0])]
            nv.append(len(r))
            raw.append(np.median(r, 0) if len(r) >= need else np.full(len(F0), np.nan))
        raw = np.array(raw)
        good = R[~np.isnan(R[:, 0])]
        if len(good) >= 3:
            s = smooth(raw)
            d.update(centres=[float(g0 + (min(win, nfr - g0) - 1) / 2) for g0 in range(0, nfr, win)],
                     raw=[None if np.isnan(x[0]) else np.round(x, 5).tolist() for x in raw], nvalid=nv,
                     resp=np.round(s, 5).tolist(), glob=np.round(np.median(good, 0), 5).tolist())
    d['secs'] = round(time.time() - t0, 1)
    if cp:
        try:
            tmp = '%s.tmp%d' % (cp, os.getpid())
            with open(tmp, 'w') as fh:
                json.dump(d, fh, separators=(',', ':'))
            os.replace(tmp, cp)
        except OSError as e:
            if log:
                log('luma EQ: table cache %s not written (%s)' % (cp, e))
    d['cached'] = None
    return d if d['resp'] is not None else None


class Table:
    """per-frame rfft-bin gains: eq_curve() of each window's smoothed response, linearly interpolated between
    window centres (held beyond the first / last centre)"""
    def __init__(self, d, ow, fmax=4.8):
        self.c = np.asarray(d['centres'], np.float64)
        self.G = np.stack([eq_curve(r, ow, fmax) for r in d['resp']]).astype(np.float32)

    def gain(self, frame):
        c = self.c
        if len(c) == 1 or frame <= c[0]:
            return self.G[0]
        if frame >= c[-1]:
            return self.G[-1]
        w = int(np.searchsorted(c, frame, 'right')) - 1
        t = np.float32((frame - c[w]) / (c[w + 1] - c[w]))
        return (1 - t) * self.G[w] + t * self.G[w + 1]

    def summary(self, fq=(1, 2, 3, 4, 5), ow=None):
        """compact log text: gain at fq MHz per window"""
        f = np.fft.rfftfreq(ow, 1 / FS)
        ix = [int(np.argmin(abs(f - q))) for q in fq]
        return self.G[:, ix]
