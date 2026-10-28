# SPDX-License-Identifier: GPL-3.0-or-later
"""Noise spectrum of a stacked TBC for the decoder's data-term weights and prior scale k.

noise_psd (v5): from the blank vertical-interval lines (field lines 6-13, samples 200-1080), on the raw TBC.
Since v6 it is robust (per-bin median over fields, fields with dropouts on those lines skipped), but it still
sees only the primary master: the stacker's cross-master blend works on the active picture, so on 3-master
sides it overstates the picture's noise (South: sigma ~1.6-1.8x).

noise_psd_active (v6 default): from the ACTIVE PICTURE of the TBC as the decoder sees it (after the deghost
inverse and the herringbone removal).  Fields k and k+8 have the same subcarrier phase (PAL 8-field
sequence), so where the picture is the same in both (held stills, black frames, flat or static areas) their
difference is noise only (x sqrt 2): disc noise is independent from frame to frame.  The difference is cut
into 16-line x 232-sample blocks; blocks around flagged dropouts are dropped; the static ones are the lowest
energy mode (selected by an iterated robust threshold); the spectrum is the per-bin median over them.

Both return the per-sample noise PSD on the rfft bins of a 928-sample decoder row (16-bit units^2 per bin,
Parseval-consistent with numpy rfft of a row)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W, H = 1135, 313
FSZ = W * H * 2


def _resample(P, n, out_len):
    """per-sample variance density on an n-point rfft grid -> |rfft|^2 of an out_len row"""
    fi = np.fft.rfftfreq(n); fo = np.fft.rfftfreq(out_len)
    return np.interp(fo, fi, P) * out_len


def _dropout_lines(tbc):
    """field index -> set of 0-based field lines with flagged dropouts (empty dict without a json)"""
    try:
        js = json.load(open(tbc + '.json'))['fields']
    except (OSError, ValueError, KeyError):
        return {}
    out = {}
    for k, fj in enumerate(js):
        d = fj.get('dropOuts')
        if d and d.get('fieldLine'):
            out[k] = {int(L) - 1 for L in d['fieldLine']}
    return out


def noise_psd(tbc, nfields=400, out_len=928, inverse=False, skip_do=True):
    """inverse: measure after the deghost inverse (as the decoder sees the TBC; v5 measured the raw TBC)"""
    if inverse:
        from deghost import channel_inverse
    fd = os.open(tbc, os.O_RDONLY)
    n = os.fstat(fd).st_size // FSZ
    dol = _dropout_lines(tbc)
    acc = []
    win = np.hanning(880)
    lo, hi = (200, n - 200) if n > 800 else (0, n - 1)     # short clips: use every field
    for fi in np.unique(np.linspace(lo, hi, min(nfields, hi - lo + 1)).astype(int)):
        if skip_do and dol.get(int(fi), set()) & set(range(6, 14)):
            continue
        f = np.frombuffer(os.pread(fd, FSZ, int(fi) * FSZ), '<u2').reshape(H, W).astype(float)
        if inverse:
            f = channel_inverse(f)
        seg = f[6:14, 200:1080]
        if np.median(seg) - np.median(f[6:14, 10:50]) < 5000:
            continue
        seg = seg - seg.mean(1, keepdims=True)
        acc.append((np.abs(np.fft.rfft(seg * win, axis=1)) ** 2).mean(0) / (win ** 2).mean())
    os.close(fd)
    if not acc:              # no usable blank-line field (v6: was a NaN spectrum); if dropouts excluded them all, use them anyway
        return noise_psd(tbc, nfields, out_len, inverse, False) if skip_do and dol else None
    # v6: per-bin median over fields (a field with a VBI glitch or a missed dropout is an outlier, not
    # averaged in); each field's value is a mean of 8 exponential bins, median/mean of Gamma(8) = 1 - 1/24
    P = np.median(acc, 0) / (1 - 1 / 24.) / 880
    return _resample(P, 880, out_len)


# ---- active picture (v6) ---------------------------------------------------------------------------
AROW0, AROW1 = 24, 308          # field lines of the picture (decoder rows LINE0 22 .. 310, minus a margin)
AX0, AX1 = 182, 1110            # decoder columns X0 .. X0 + 928
BY, BX = 16, 232                # block: field lines x samples
LAG = 8                         # fields: same subcarrier phase and field parity
NBAND = 7                       # bins (of BX) per selection band
_bwin = np.hanning(BX)


def _pair_blocks(args):
    """one field pair -> (temporal spectra of (f_k - f_k+LAG) / sqrt 2, spatial spectra of f_k (each line minus
    its straight-line fit), block mean levels of f_k), per 16 x 232 block"""
    tbc, k, bad = args
    from deghost import channel_inverse
    from deherring import deherring_field
    fd = os.open(tbc, os.O_RDONLY)
    f = [np.frombuffer(os.pread(fd, FSZ, (k + d) * FSZ), '<u2').reshape(H, W) for d in (0, LAG)]
    os.close(fd)
    g = [deherring_field(channel_inverse(x))[0] for x in f]
    d = (g[0] - g[1])[AROW0:AROW1, AX0:AX1] / np.sqrt(2)
    a = g[0][AROW0:AROW1, AX0:AX1]
    t = np.arange(BX) - (BX - 1) / 2.
    St, Ss, L = [], [], []
    for r0 in range(0, d.shape[0] - BY + 1, BY):
        if any((AROW0 + r) in bad for r in range(r0 - 1, r0 + BY + 1)):
            continue
        for c0 in range(0, d.shape[1] - BX + 1, BX):
            b = d[r0:r0 + BY, c0:c0 + BX]
            b = b - b.mean(1, keepdims=True)
            St.append((np.abs(np.fft.rfft(b * _bwin, axis=1)) ** 2).mean(0))
            q = a[r0:r0 + BY, c0:c0 + BX]
            L.append(q.mean())
            q = q - q.mean(1, keepdims=True); q = q - np.outer((q * t).sum(1) / (t * t).sum(), t)
            Ss.append((np.abs(np.fft.rfft(q * _bwin, axis=1)) ** 2).mean(0))
    z = np.zeros((0, BX // 2 + 1))
    return (np.array(St).reshape(-1, BX // 2 + 1) if St else z, np.array(Ss).reshape(-1, BX // 2 + 1) if Ss else z,
            np.array(L))


def _lowest_mode(S):
    """per band (NBAND bins ~0.54 MHz) the mean spectrum of the lowest-energy mode of the blocks (see below)"""
    # Start from the 2nd percentile, then iterate: centre = median of the selected, keep blocks below
    # centre * (1 + 3 CV), CV = 1.5 x the theoretical spread of a noise-only block's band energy (BY lines x
    # NBAND Hann-windowed bins; adjacent-bin |corr|^2 4/9 and 1/36).  Spectrum = mean over the selected.
    nb = NBAND
    cvt = 1.5 * np.sqrt((nb + 2 * (nb - 1) * 4 / 9. + 2 * (nb - 2) / 36.) / (BY * nb * nb))
    P = np.zeros(S.shape[1]); used = []
    for b0 in range(1, S.shape[1], nb):
        Eb = S[:, b0:b0 + nb].sum(1)
        p2 = np.percentile(Eb, 2); sel = Eb <= p2
        for _ in range(12):
            # a real noise mode settles at median ~ p2 / (1 - 2 CV); without one (every pair differs, e.g. a
            # clip of CAV stills) the median would creep up through the content: capped there
            new = Eb <= min(np.median(Eb[sel]), 1.6 * p2) * (1 + 3 * cvt)
            if (new == sel).all():
                break
            sel = new
        P[b0:b0 + nb] = S[sel, b0:b0 + nb].mean(0); used.append(int(sel.sum()))
    P[0] = P[1]                                  # rows are mean-removed
    return P / (_bwin ** 2).mean() / BX, used


def _black(tbc):
    try:
        return float(json.load(open(tbc + '.json'))['videoParameters']['black16bIre'])
    except (OSError, ValueError, KeyError):
        return 16384.0


def noise_psd_active(tbc, npairs=200, out_len=928, workers=8, info=None, parts=False):
    """see the module doc.  info: optional dict, filled with diagnostics.  parts: also return the temporal and
    spatial estimates"""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    n = os.path.getsize(tbc) // FSZ
    if n <= LAG:
        return None
    dol = _dropout_lines(tbc)
    ks = np.unique(np.linspace(0, n - 1 - LAG, min(npairs, n - LAG)).astype(int))
    jobs = [(tbc, int(k), dol.get(int(k), set()) | dol.get(int(k) + LAG, set())) for k in ks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as ex:
        res = list(ex.map(_pair_blocks, jobs))
    St = np.concatenate([r[0] for r in res]); Ss = np.concatenate([r[1] for r in res]); L = np.concatenate([r[2] for r in res])
    # Flattened blocks are left out: an older stacker returned held black as a near-constant signal (North panel:
    # sigma 60 from them, the priors ~6x too weak).  Analog capture noise cannot vanish, so a block whose spatial
    # 5.5-8 MHz energy is below FLAT_FRAC x the median block's is not noise of the picture.
    fb = np.fft.rfftfreq(BX, 1 / 17.734475); hb = (fb > 5.5) & (fb < 8.0)
    eh = Ss[:, hb].sum(1)
    keep = eh > FLAT_FRAC * np.median(eh)
    St, Ss = St[keep], Ss[keep]
    if len(St) < 20:          # (nearly) every block excluded (dropouts, black)
        return None
    Pt, used = _lowest_mode(St)
    Ps, _ = _lowest_mode(Ss)
    # The temporal estimate assumes independent noise from frame to frame (true of disc and player noise); a
    # stack that smooths a held run over time breaks that and it reads too low (2026-10-27: a candidate stacker
    # gave sigma 131 against 697, priors 5x weaker, maps SOFTER).  So it is bounded below by SPATIAL_FLOOR x the
    # spatial estimate (the same lowest-mode selection on single fields, each line minus its straight-line fit:
    # an upper bound where the flattest blocks carry some picture, the true noise where they are flat).
    P = np.maximum(Pt, SPATIAL_FLOOR * Ps)
    if info is not None:
        info.update(blocks=int(len(St)), flattened=int((~keep).sum()), used_min=min(used), used_max=max(used), pairs=len(jobs),
                    temporal=round(noise_sigma(_resample(Pt, BX, out_len))), spatial=round(noise_sigma(_resample(Ps, BX, out_len))))
    if parts:
        return _resample(P, BX, out_len), _resample(Pt, BX, out_len), _resample(Ps, BX, out_len)
    return _resample(P, BX, out_len)


SPATIAL_FLOOR = float(os.environ.get('PALDEC_NOISE_SFLOOR', 0.7))
FLAT_FRAC = float(os.environ.get('PALDEC_NOISE_FLAT', 0.1))


def combine(Pa, Pv, fs=17.734475, out_len=928):
    """active-picture estimate Pa, guarded by the blank-line one Pv (both after the inverse, 928-row bins).
    The picture's noise cannot exceed the blank lines' (the blend only removes noise), and above 5.5 MHz the
    discs carry no picture, so there Pa is never content: q = Pa/Pv (smoothed over ~0.5 MHz) is capped at
    min(1, QCAP x its median over 5.5-8 MHz); QCAP 1.75: on segments with static content q / that median is
    0.8-1.7 (largest at 2-3.5 MHz, where less of the noise is shared by a master's pressings).  Where Pa has caught picture content (a clip of different stills:
    every pair differs) the spectrum becomes Pv scaled by the clean high band's ratio, not a mix of the two
    shapes (a per-bin minimum: LF from the blank lines, HF from the picture, which shifts the normalised data
    weights: fq benchmark stills segment -0.8 dB luma)."""
    k = np.ones(27) / 27
    sm = lambda P: np.convolve(np.pad(P, 13, mode='edge'), k, 'valid')
    q = sm(Pa) / np.maximum(sm(Pv), 1e-9)
    f = np.fft.rfftfreq(out_len, 1 / fs)
    qh = float(np.median(q[(f > 5.5) & (f < 8.0)]))
    return Pv * np.minimum(q, min(1.0, QCAP * qh)), qh


QCAP = float(os.environ.get('PALDEC_NOISE_QCAP', 1.75))   # v6_0: 2
AGREE = 0.3


LAST_CAPPED = None


def estimate(tbc, mode='active', log=None):
    """the decoder's noise spectrum of tbc -> (P, message).  mode 'vbi': the blank lines on the raw TBC (v5);
    'active': noise_psd_active guarded by the blank lines (both after the inverse).  Either estimator failing
    (no usable field or block) falls back to the other."""
    if mode == 'vbi':
        P = noise_psd(tbc, nfields=200)
        if P is None:
            raise SystemExit('noise: no usable blank-line field in ' + tbc)
        return P, 'noise: blank lines (raw) sigma %.0f' % noise_sigma(P)
    info = {}
    r = noise_psd_active(tbc, info=info, parts=True); Pv = noise_psd(tbc, nfields=200, inverse=True)
    Pa = r[0] if r is not None else None
    if Pa is None and Pv is None:
        raise SystemExit('noise: neither the picture nor the blank lines of %s give a noise estimate' % tbc)
    if Pa is None:
        return Pv, 'noise: active estimate failed (too few fields / blocks): blank lines, sigma %.0f' % noise_sigma(Pv)
    if Pv is None:
        return Pa, 'noise: no usable blank-line field: active estimate unguarded, sigma %.0f' % noise_sigma(Pa)
    P, qh = combine(Pa, Pv)
    # where the temporal and the spatial estimates agree (two independent measurements of the same flat,
    # static blocks: within AGREE in power, 0.5 MHz smoothed) the active spectrum is the picture's noise and
    # the cap above (made for clips without static content) does not apply; only the blank lines bound it.
    # North v8 side: temporal 505 / spatial 514, the cap alone gave 223
    sm = lambda X: np.convolve(np.pad(X, 13, mode='edge'), np.ones(27) / 27, 'valid')
    t, sp = sm(r[1]), sm(r[2])
    agree = np.abs(t - sp) <= AGREE * np.maximum(t, sp)
    global LAST_CAPPED
    LAST_CAPPED = P                     # the capped combination (v6_0's choice), before the agreement override
    P = np.where(agree, np.minimum(Pa, Pv), P)
    info['agree'] = round(float(agree.mean()), 2)
    return P, 'noise: active sigma %.0f, blank lines %.0f, used %.0f (5.5-8 MHz power ratio %.3f)  %s' % (
        noise_sigma(Pa), noise_sigma(Pv), noise_sigma(P), qh, info)


def noise_sigma(P, out_len=928):
    """rms noise (16-bit) of a spectrum from noise_psd*: DC and Nyquist bins count once, the others twice"""
    return float(np.sqrt((2 * P.sum() - P[0] - P[-1]) / out_len ** 2))


if __name__ == '__main__':
    f = np.fft.rfftfreq(928, 1 / 17.734475)
    for nm, fn in (('vbi', noise_psd), ('active', noise_psd_active)):
        info = {}
        P = fn(sys.argv[1], inverse=True) if nm == 'vbi' else fn(sys.argv[1], info=info)
        if len(sys.argv) > 2:
            np.save(sys.argv[2] + '.' + nm + '.npy', P)
        print('%-6s noise std %.1f (16-bit); share above 5.5 MHz %.0f%%  %s' % (nm, noise_sigma(P), 100 * P[f > 5.5].sum() / P.sum(), info))
