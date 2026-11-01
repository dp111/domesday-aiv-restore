# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-TBC luma equaliser from the disc's own multiburst VITS (v6_3, 2026-10-31).

The deghost inverse G (deghost.py) lifts everything above ~1 MHz by 16-18 % over DC.  The disc's CCIR
multiburst (0.5 / 1 / 2 / 4 / 4.8 / 5.8 MHz packets between a 2-level reference bar; Domesday: field 0
line 20) measures the whole record/playback chain: on Community South the raw stack reads 0.91 / 0.92 /
0.93 / 1.00 / 0.90 / 0.51 of the reference, so after G the picture's luma is 1.00 / 1.06 / 1.08 / 1.17 /
1.04 at 0.5-4.8 MHz (other discs +4 .. +27 %); its 4.43 MHz chroma reads the same lift (transform3d 1.094
on the EBU bars).  The equaliser is 1 / that response (pchip through DC = 1 and the packets, held above
4.8 MHz), never a boost (<= 1) and >= EQ_FLOOR, applied to the decoded luma rows only (chroma / the solve
untouched).  It is measured once per TBC from frames spread over the WHOLE file, so chunked runs agree.
No multiburst found (synthetic bench, other VITS) -> no equaliser.
"""
import json, os
import numpy as np

FS = 17.734475
F0 = (0.5, 1.0, 2.0, 4.0, 4.8, 5.8)
SEG = ((425, 495), (535, 610), (650, 725), (758, 840), (862, 950), (972, 1062))
REF_HI, REF_LO = (230, 268), (290, 340)
EQ_FLOOR = float(os.environ.get('PALDEC_LUMA_EQ_FLOOR', 0.75))


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
