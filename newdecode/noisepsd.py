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


def noise_psd(tbc, nfields=400, out_len=928, inverse=False):
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
        if dol.get(int(fi), set()) & set(range(6, 14)):
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
    """one field pair -> block spectra (nb, BX//2+1) of (f_k - f_k+LAG) / sqrt 2"""
    tbc, k, bad = args
    from deghost import channel_inverse
    from deherring import deherring_field
    fd = os.open(tbc, os.O_RDONLY)
    f = [np.frombuffer(os.pread(fd, FSZ, (k + d) * FSZ), '<u2').reshape(H, W) for d in (0, LAG)]
    os.close(fd)
    g = [deherring_field(channel_inverse(x))[0] for x in f]
    d = (g[0] - g[1])[AROW0:AROW1, AX0:AX1] / np.sqrt(2)
    S = []
    for r0 in range(0, d.shape[0] - BY + 1, BY):
        if any((AROW0 + r) in bad for r in range(r0 - 1, r0 + BY + 1)):
            continue
        for c0 in range(0, d.shape[1] - BX + 1, BX):
            b = d[r0:r0 + BY, c0:c0 + BX]
            b = b - b.mean(1, keepdims=True)
            s = (np.abs(np.fft.rfft(b * _bwin, axis=1)) ** 2).mean(0)
            S.append(s)
    return None, np.array(S).reshape(-1, BX // 2 + 1)


def noise_psd_active(tbc, npairs=200, out_len=928, workers=8, info=None):
    """see the module doc.  info: optional dict, filled with diagnostics (blocks, used, cv)."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    n = os.path.getsize(tbc) // FSZ
    dol = _dropout_lines(tbc)
    ks = np.unique(np.linspace(0, n - 1 - LAG, min(npairs, n - LAG)).astype(int))
    jobs = [(tbc, int(k), dol.get(int(k), set()) | dol.get(int(k) + LAG, set())) for k in ks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as ex:
        res = list(ex.map(_pair_blocks, jobs))
    S = np.concatenate([r[1] for r in res])
    # Per band (NBAND bins ~0.54 MHz): the static blocks are the lowest-energy mode of the blocks' band energy.
    # Per band, because a smooth area of two DIFFERENT pictures (CAV stills) differs at low frequencies only.
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
    P = P / (_bwin ** 2).mean() / BX
    if info is not None:
        info.update(blocks=int(len(S)), used_min=min(used), used_max=max(used), pairs=len(jobs))
    return _resample(P, BX, out_len)


def combine(Pa, Pv, fs=17.734475, out_len=928):
    """active-picture estimate Pa, guarded by the blank-line one Pv (both after the inverse, 928-row bins).
    The picture's noise cannot exceed the blank lines' (the blend only removes noise), and above 5.5 MHz the
    discs carry no picture, so there Pa is never content: q = Pa/Pv (smoothed over ~0.5 MHz) is capped at
    min(1, 2 x its median over 5.5-8 MHz).  Where Pa has caught picture content (a clip of different stills:
    every pair differs) the spectrum becomes Pv scaled by the clean high band's ratio, not a mix of the two
    shapes (a per-bin minimum: LF from the blank lines, HF from the picture, which shifts the normalised data
    weights: fq benchS4 stills segment -0.8 dB luma)."""
    k = np.ones(27) / 27
    sm = lambda P: np.convolve(np.pad(P, 13, mode='edge'), k, 'valid')
    q = sm(Pa) / np.maximum(sm(Pv), 1e-9)
    f = np.fft.rfftfreq(out_len, 1 / fs)
    qh = float(np.median(q[(f > 5.5) & (f < 8.0)]))
    return Pv * np.minimum(q, min(1.0, 2 * qh)), qh


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
