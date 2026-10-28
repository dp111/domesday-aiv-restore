# SPDX-License-Identifier: GPL-3.0-or-later
"""Remove the disc's FM-beat herringbone from TBC fields.

Mechanism (2026-10-27, measured on the EBU bars of Community South / North and on single captures of
each glass master): the beat is a tone at f_c - f_sc, f_c the instantaneous FM video carrier.  Its
frequency follows the colour's luma (2.72 MHz blue .. 3.19 yellow), its amplitude the chroma, and its
phase is exactly that of e^{i(Phi(t) - phi_sc(t))}: Phi the FM carrier phase (integral of the
carrier frequency of the PRE-EMPHASISED composite, 7.088 MHz at black + 8 kHz/IRE), phi_sc the
chroma's own phase.  A physical FM simulation with 2nd-harmonic distortion of the carrier (pit
asymmetry: the component at 2f_c - f_sc beats with the carrier) reproduces frequency, colour
dependence and phase; a baseband-chroma leak into the RF gives the same law.  It is per glass master
(same on every capture of it, -15/-16 dB removable on a single capture), so a stack keeps it.  Line to
line its phase steps by a near-constant amount and wanders slowly (+-100 deg over ~25 lines: the
source's line timing), so it cannot be predicted from one line to the next exactly.

deherring_field (model canceller): per field, build the predicted beat waveform B(t) from the field
itself (de-emphasised, 2.2-3.9 MHz), fit one complex amplitude per line by least squares within a
Hann window of SM lines after removing the tracked line-to-line phase, and subtract Re(c * B).
  * the fit uses only flat parts of the picture (fit weight falls off at luma / chroma-envelope edges:
    map lines, text and colour edges cannot pull the amplitude); the fitted beat is subtracted
    everywhere along the line, edges included (it is the physics, not a filter);
  * a window is corrected only when its amplitude is well determined (|c| / standard error > Z_MIN,
    full correction from 2 Z_MIN): fields without flat saturated colour (maps, most photos) are left
    exactly as they were.
PALDEC_DEHERRING: model (default) | tiles (the Oct 12 flat-tile remover, deherring_tiles) | both | 0.
A stack that blends several masters carries several beats with independent phase: one fit removes
less there (North panel bars -5.6 dB, South 4-cap -11 dB, single master -15 dB) - best done per master
in the stacker.

Oct 12 tile remover (deherring_tiles):Remove the disc's FM-beat herringbone from TBC fields.

Community South carries a beat between the FM video carrier and the colour
signal that was recorded on the master (it survives a 9-capture stack): in
strongly coloured flat areas a faint (~1 level peak, 8-bit) slanted plane
wave at horizontal frequency
    f = F0 + K * brightness   (F0 2.232 MHz, K 2.34e-5 MHz per 16-bit level;
                               fitted over 300 detections, 97 kHz rms)
- 2.8-3.2 MHz at the usual levels: herringbone in reds and magentas.

It is far below the noise on any one line, so it is estimated coherently
over 2-D tiles (TY field lines x TX samples): each line is demodulated at
the predicted frequency, the tile's line-to-line phase step is found from
the spectrum of those complex amplitudes, and one amplitude/phase is fitted
for the whole tile.  A tile is corrected only when
  * it is strongly coloured (the beat needs colour),
  * the plane wave holds most of the demodulated energy across its lines
    (coherent, not noise or texture),
  * its line-to-line step is not ~0 (Test Card F's gratings and other
    vertical structure are line-locked),
  * the picture is flat there (<1.5 MHz luma std under FLAT_MAX) and the
    wave is faint (under AMP_MAX) - textured areas can contain a
    coincidentally coherent slanted component that is real detail.
Correction is the fitted wave times a 2-D Hann, overlap-added (50%), so
tiles that fail the tests are left exactly as they were."""
import os
import numpy as np, scipy.fft as sfft, scipy.signal as sps
from scipy.ndimage import uniform_filter1d

MODE = os.environ.get('PALDEC_DEHERRING', 'model')
STAMP = MODE + ':model-2026-10-27b'     # newdecode's de-smear cache key
FS = 17.734475
F0_HZ = float(os.environ.get('PALDEC_DH_F0', 7.088e6))   # FM carrier at black (bars fit: South A+B, North A 7.088;
HZ_IRE = 8000.0                   # North B 7.079 MHz; other discs: deherring.py calibrate) and per IRE
GAMMA = 0.4                       # weight of the chroma's own PM in the carrier phase (fitted)
ALPHA = -1.0                      # beat amplitude law exp(ALPHA * (f_c - 7.5 MHz)): falls with brightness
BAND = (2.2e6, 3.9e6)
SM = 15                           # lines per amplitude window
TRACK = 9                         # lines over which the line-to-line phase advance is tracked
EDGE = 6000.0                     # 16-bit; fit weight 1 / (1 + (g / EDGE)^4), g = 25-sample sum of |luma'| + |envelope'|
Z_MIN = 4.0
C_MAX = float(os.environ.get('PALDEC_DH_CMAX', 450.0))   # largest plausible beat amplitude |c| (basis units)
EDGE_SUB = float(os.environ.get('PALDEC_DH_EDGE_SUB', 0))   # optional subtraction taper at edges (same g as EDGE), e.g. 12000; 0 = off (known truth: at edges the untapered beat removal is right, -5..-9 dB)
E_FIT, E_SUB = 2500.0 ** 2, 3500.0 ** 2   # 2.2-3.9 MHz band power (25-sample mean) above which no fit / no subtraction
PRE_CHROMA, PRE_FLAT, PRE_COUNT = 4000.0, 1500.0, 1500   # pre-gate: skip fields with < PRE_COUNT flat coloured samples
W, H = 1135, 313
_FSC = 4.43361875e6; _FS = FS * 1e6


def _emph(t1, t2):
    w1 = 2 * _FS * np.tan((1 / t1) / (2 * _FS)); w2 = 2 * _FS * np.tan((1 / t2) / (2 * _FS))
    b, a = sps.zpk2tf([-w1], [-w2], w2 / w1)
    return sps.bilinear(b, a, _FS)


_DEEMP = _emph(100e-9, 400e-9); _PREEMP = _emph(400e-9, 100e-9)   # ld-decode PAL video_deemp
_N = 368640                       # >= W*H + guard, 2^13 * 45: fast
_fr = sfft.fftfreq(_N, 1 / _FS)
_rf = sfft.rfftfreq(_N, 1 / _FS)
_CB = (2 * np.clip((_rf - 3.3e6) / 0.4e6, 0, 1) * np.clip((5.6e6 - _rf) / 0.4e6, 0, 1)).astype(np.float32)  # analytic chroma
_bp = lambda a: np.clip((a - BAND[0]) / 0.3e6 + 0.5, 0, 1) * np.clip((BAND[1] - a) / 0.3e6 + 0.5, 0, 1)
_BP = _bp(np.abs(_fr)).astype(np.float32); _BPR = _bp(_rf).astype(np.float32)
_BW_CORR = (_FS / 2) / (BAND[1] - BAND[0])
_HP = (sps.freqz(*_PREEMP, worN=[_FSC], fs=_FS)[1][0])             # pre-emphasis at the subcarrier


def _box(v, n):
    return uniform_filter1d(v, n, mode='nearest')


def _levels():
    import palmodel as M
    return M.P['AY'] * 4096 + M.P['B'], M.P['AY'] * 60160 + M.P['B']


def _beat_basis(x, black, white, f0=None):
    """predicted beat (complex, 2.2-3.9 MHz, de-emphasised), fit weights and the band-passed field, flattened"""
    ire = (x - black) * (100.0 / (white - black))
    ip = sps.lfilter(*_PREEMP, ire - ire[0]) + ire[0]
    X = sfft.rfft(x.astype(np.float32), _N)
    sp = np.zeros(_N, np.complex64); sp[:X.size] = X * _CB
    ca = sfft.ifft(sp)[:x.size]                                       # analytic chroma
    xb = sfft.irfft(X * _BPR, _N)[:x.size]                            # the field, 2.2-3.9 MHz
    cp = (ca * _HP).real * (100.0 / (white - black))                  # its pre-emphasised part of ip
    fc = (F0_HZ if f0 is None else f0) + HZ_IRE * (ip - (1 - GAMMA) * cp)
    ph = np.cumsum(fc * (2 * np.pi / _FS)) % (2 * np.pi)
    lf = _box(_box(fc, 4), 17)                                        # carrier without the chroma PM (box 4 nulls f_sc)
    ph = ph.astype(np.float32); am = ((lf - _FSC) / 3e9 * np.exp(ALPHA * (lf - 7.5e6) / 1e6)).astype(np.float32)
    r = (np.cos(ph) * am + 1j * (np.sin(ph) * am)) * np.conj(ca)
    r = (sps.lfilter(*_DEEMP, r.real) + 1j * sps.lfilter(*_DEEMP, r.imag)).astype(np.complex64)
    rb = sfft.ifft(sfft.fft(r, _N) * _BP)[:x.size]
    lum = _box(_box(x - ca.real, 4), 9); env = _box(np.abs(ca), 17)
    g = _box(np.abs(np.gradient(lum)).astype(np.float32) + np.abs(np.gradient(env)), 25) * 25
    # in-band picture energy: gratings / fine texture at 2-4 MHz have no low-frequency edge but far more band
    # power than any beat (<= ~1500 peak): no fit there, and no subtraction where it is well above it
    e = _box(xb * xb, 25)
    wt = 1 / (1 + (g / EDGE) ** 4) / (1 + (e / E_FIT) ** 2)
    ms = 1 / (1 + (e / E_SUB) ** 2)
    if EDGE_SUB:      # leave strong luma / colour edges as they were (decoder edge decisions; the beat is masked there)
        ms = ms * _box(1 / (1 + (g / EDGE_SUB) ** 4), 9)
    return rb, wt, xb, ms


def deherring_field(field, l0=24, l1=308, x0=185, x1=1107, levels=None, info=None, f0=None):
    """Library API (stable; newdecode and the stacker use it).
    field: one TBC field (313 x 1135) AFTER deghost.channel_inverse (the de-smeared signal the beat model is fitted on);
    levels: (black16bIre, white16bIre) of the TBC (default: palmodel.P as set by newdecode); f0: this master's FM
    carrier at black in Hz (default F0_HZ / PALDEC_DH_F0; per side or per master from calibrate()).
    -> (field with the beat removed, float64; number of lines corrected).  MODE (PALDEC_DEHERRING) selects the remover;
    info (dict) receives gate / c / th per line.  Fields without flat colour come back unchanged (same values)."""
    if MODE == '0':
        return np.asarray(field, np.float64), 0
    if MODE == 'tiles':
        return deherring_tiles(field, l0, l1)
    f = np.asarray(field, np.float64)
    black, white = levels or _levels()
    # quick pre-gate: the beat needs flat saturated colour; x[n] - x[n+2] is twice the chroma at 4fsc
    a = f[l0:l1, x0:x1]
    ch = uniform_filter1d(np.abs(a[:, 2:] - a[:, :-2]), 8, axis=1) * (np.pi / 4)   # ~ chroma amplitude
    lu = uniform_filter1d(a, 4, axis=1)
    fl = uniform_filter1d(np.abs(lu[:, 8:] - lu[:, :-8]), 16, axis=1)              # luma change over 8 samples
    if np.count_nonzero((ch[:, :fl.shape[1]] > PRE_CHROMA) & (fl < PRE_FLAT)) < PRE_COUNT:
        if info is not None:
            info.update(gate=np.zeros(l1 - l0), skipped=True)
        return (deherring_tiles(f, l0, l1) if MODE == 'both' else (f, 0))
    rb, wt, xb, ms = _beat_basis(f.reshape(-1), black, white, f0)
    B = rb.reshape(f.shape)
    d = xb.reshape(f.shape)[l0:l1, x0:x1].astype(np.float64)
    w = wt.reshape(f.shape)[l0:l1, x0:x1]
    Bs = B[l0:l1, x0:x1]
    b1, b2 = Bs.real, -Bs.imag                                        # Re(c B) = cr b1 + ci b2
    g11 = (w * b1 * b1).sum(1); g22 = (w * b2 * b2).sum(1); g12 = (w * b1 * b2).sum(1)
    r1 = (w * b1 * d).sum(1); r2 = (w * b2 * d).sum(1)
    det = g11 * g22 - g12 ** 2 + 1e-9 * (g11 + g22) ** 2 + 1e-12
    z = ((g22 * r1 - g12 * r2) + 1j * (g11 * r2 - g12 * r1)) / det    # per-line amplitude
    wl = g11 + g22
    # tracked phase: angle of the weighted, smoothed z_l z*_(l-1), integrated over the lines
    u = np.concatenate([[0], z[1:] * np.conj(z[:-1]) * np.sqrt(wl[1:] * wl[:-1])])
    th = np.cumsum(np.angle(np.convolve(u, np.hanning(TRACK + 2)[1:-1], 'same'))); th -= th[0]
    e = np.exp(1j * th)
    er, ei = e.real, e.imag
    # c_l = s * e_l with s constant over the window: rotated normal equations, Hann-weighted over SM lines
    a11 = er * er * g11 + 2 * er * ei * g12 + ei * ei * g22
    a22 = ei * ei * g11 - 2 * er * ei * g12 + er * er * g22
    a12 = -er * ei * g11 + (er * er - ei * ei) * g12 + er * ei * g22
    q1 = er * r1 + ei * r2; q2 = -ei * r1 + er * r2
    k = np.hanning(SM + 2)[1:-1]
    A11, A22, A12, Q1, Q2 = (np.convolve(v, k, 'same') for v in (a11, a22, a12, q1, q2))
    dt = A11 * A22 - A12 ** 2 + 1e-12
    s = ((A22 * Q1 - A12 * Q2) + 1j * (A11 * Q2 - A12 * Q1)) / dt
    # confidence gate: |s| over its standard error (noise from the weighted band power, conservative)
    sig2 = (w * d * d).sum() / max(w.sum(), 1.0)
    # (x BW_CORR: band-limited noise is correlated over ~FS/2/bandwidth samples, the plain LS formula assumes white)
    se = np.sqrt(sig2 / np.maximum((A11 + A22) / 2, 1e-12) * ((k ** 2).sum() / k.sum()) * _BW_CORR)
    gate = np.clip((np.abs(s) / se - Z_MIN) / Z_MIN, 0, 1)
    # physical bound: the beat constant is a disc property (|c| 135-270 on bars, maps, photos of South / North);
    # a far larger fit is picture structure or a defect that happens to correlate (s29696 frame 220: 1112)
    gate = gate * np.clip((C_MAX * 1.25 - np.abs(s)) / (C_MAX * 0.25), 0, 1)
    c = s * gate * e
    out = f.copy()
    on = gate > 0
    if on.any():
        out[l0:l1] -= (c[:, None] * B[l0:l1]).real * ms.reshape(f.shape)[l0:l1]
    if info is not None:
        info.update(gate=gate, c=c, th=th)
    if MODE == 'both':
        out, n2 = deherring_tiles(out, l0, l1)
    return out, int(on.sum())


# ---- Oct 12 flat-tile remover

F0, K = 2.232, 2.34e-5
TX, TY = 64, 32
CHROMA_MIN = 15000.0
COH_MIN = 0.35        # fraction of the tile's demodulated energy in the plane wave
FLAT_MAX = 3.0 * 256  # 16-bit std of the tile's <1.5 MHz luma: herringbone only where the picture is flat
AMP_MAX = 4.0 * 256   # the beat is a faint 1-3 levels; a stronger 'wave' is picture texture
STEP_MIN = 0.08       # cycles per field line: not line-locked
X0, X1 = 186, 1106
_wx = np.hanning(TX); _wy = np.hanning(TY)
_fq = np.fft.rfftfreq(256, 1 / FS); _chr = (_fq > 3.9) & (_fq < 5.0)


def deherring_tiles(field, l0=24, l1=308):
    f = field.astype(np.float64)
    corr = np.zeros_like(f); n = 0
    xs = np.arange(TX)
    for y in range(l0, l1 - TY + 1, TY // 2):
        for x in range(X0, X1 - TX + 1, TX // 2):
            t = f[y:y + TY, x:x + TX]
            lvl = t.mean(); d = t - t.mean(1, keepdims=True)
            # flat-area gate (frame 16275's dense floral texture once lost
            # 21 levels of real detail to a coincidentally coherent 'wave')
            lo = np.fft.irfft(np.fft.rfft(t, axis=1) * (np.fft.rfftfreq(TX, 1 / FS) < 1.5), TX, axis=1)
            if lo.std() > FLAT_MAX:
                continue
            P = (np.abs(np.fft.rfft(d * _wx, 256, axis=1)) ** 2).mean(0)
            chroma = np.sqrt(P[_chr].sum()) * 2 / _wx.sum()
            if chroma < CHROMA_MIN:
                continue
            fp = F0 + K * lvl
            e = np.exp(-2j * np.pi * fp / FS * xs)
            z = (d * _wx * e).sum(1) / _wx.sum() * 2           # complex amplitude per line
            Z = np.fft.fft(z * _wy, 512)
            k = int(np.argmax(np.abs(Z))); fy = np.fft.fftfreq(512)[k]
            if abs(fy) < STEP_MIN:
                continue
            ly = np.arange(TY)
            A = (z * np.exp(-2j * np.pi * fy * ly) * _wy).sum() / _wy.sum()
            model = A * np.exp(2j * np.pi * fy * ly)
            coh = (np.abs(model) ** 2 * _wy).sum() / ((np.abs(z) ** 2 * _wy).sum() + 1e-9)
            if coh < COH_MIN or abs(A) > AMP_MAX:
                continue
            wave = np.real(model[:, None] * np.exp(2j * np.pi * fp / FS * xs)[None, :])
            corr[y:y + TY, x:x + TX] += wave * _wy[:, None] * _wx[None, :]
            n += 1
    return f - corr, n


def cancel_tbc_field(field, black, white, f0=None):
    """Library API for TBC-domain users (the stacker): raw TBC field (as stored, channel smear included) -> the same
    field with the beat removed, still in the TBC domain (float64): x - H(removed), H = the channel smear deghost
    inverts, so a later channel_inverse sees exactly deherring_field's output.  -> (field, lines corrected)"""
    from deghost import channel_inverse
    from scipy.signal import lfilter
    x = np.asarray(field, np.float64)
    g = channel_inverse(x)
    o, n = deherring_field(g, levels=(black, white), f0=f0)
    if not n:
        return x, 0
    c, tau = 0.16, 9.0; a = np.exp(-1.0 / tau)                        # deghost.deghost's channel (exact inverse)
    r = (g - o).reshape(-1)
    return x - lfilter([1.0 - c * a, -(1.0 - c) * a], [1.0, -a], r).reshape(x.shape), n


def calibrate(tbc, nfields=48, grid=np.arange(7.064e6, 7.1121e6, 0.003e6)):
    """Library API: F0 (Hz) for one TBC (a side's stack, or one capture = one master): the carrier that lets the
    canceller remove the most coherent beat energy on the nfields fields with the most flat colour.  Returns the
    best f0 (also printed); does not change F0_HZ.  CLI: python deherring.py calibrate X.tbc [nfields]"""
    import json, sys
    from deghost import channel_inverse
    vp = json.load(open(tbc + '.json'))['videoParameters']; lv = (vp['black16bIre'], vp['white16bIre'])
    mm = np.memmap(tbc, np.uint16, 'r'); nf = mm.size // (W * H)
    cand = []
    for k in np.linspace(0, nf - 1, min(nf, 600)).astype(int):        # pre-gate count on a spread of fields
        a = np.asarray(mm[k * W * H:(k + 1) * W * H], np.float64).reshape(H, W)[24:308, 185:1107]
        ch = uniform_filter1d(np.abs(a[:, 2:] - a[:, :-2]), 8, axis=1) * (np.pi / 4)
        lu = uniform_filter1d(a, 4, axis=1); fl = uniform_filter1d(np.abs(lu[:, 8:] - lu[:, :-8]), 16, axis=1)
        cand.append((np.count_nonzero((ch[:, :fl.shape[1]] > PRE_CHROMA) & (fl < PRE_FLAT)), k))
    fields = [k for c, k in sorted(cand)[::-1][:nfields] if c >= PRE_COUNT]
    F = [channel_inverse(np.asarray(mm[k * W * H:(k + 1) * W * H], np.float64).reshape(H, W)) for k in fields]
    res = []
    for f0 in grid:
        res.append(sum(float(((f - deherring_field(f, levels=lv, f0=f0)[0]) ** 2).sum()) for f in F))
        print('F0 %.3f MHz: removed energy %.4g' % (f0 / 1e6, res[-1]), file=sys.stderr, flush=True)
    best = grid[int(np.argmax(res))]
    print('%d fields; best F0 %.3f MHz (PALDEC_DH_F0=%.0f)' % (len(F), best / 1e6, best))
    return best


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == 'calibrate':
        calibrate(sys.argv[2], *(int(x) for x in sys.argv[3:4]))
