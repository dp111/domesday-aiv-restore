# SPDX-License-Identifier: GPL-3.0-or-later
"""Measurements shared by the panel builder and the scorer (composite PAL fields, 4fsc sampling)."""
import numpy as np
import scipy.fft as sfft

FS = 17734475.0
H, W = 313, 1135
RA, CA = slice(24, H - 6), slice(185, 1107)       # stack.py _active()
BANDS = ((0.0, 2.0), (2.0, 3.5), (3.5, 5.5), (5.5, FS / 2e6))
BAND_NAMES = ("0-2", "2-3.5", "3.5-5.5", "5.5+")
BLK_H, BLK_W = 16, 64


def band_masks(n):
    f = np.fft.rfftfreq(n, 1.0 / FS) / 1e6
    return [(f >= a) & (f < b) if i < len(BANDS) - 1 else (f >= a) for i, (a, b) in enumerate(BANDS)]


_BM = {}


def _bm(n):
    if n not in _BM:
        m = band_masks(n)
        wgt = np.full(n // 2 + 1, 2.0)
        wgt[0] = 1.0
        if n % 2 == 0:
            wgt[-1] = 1.0
        _BM[n] = (m, wgt)
    return _BM[n]


def band_ms(e):
    """Mean-square of e (rows x n) per band, along rows; the bands sum to mean(e**2)."""
    e = np.asarray(e, np.float64)
    n = e.shape[-1]
    m, wgt = _bm(n)
    P = (np.abs(sfft.rfft(e, axis=-1, workers=1)) ** 2 * wgt).reshape(-1, n // 2 + 1).sum(axis=0)
    tot = e.size * n
    return np.array([P[mk].sum() / tot for mk in m])


def band_cross(a, b):
    """Per band <a,b> and <b,b> (rows x n) for detail gain a vs reference b."""
    n = a.shape[-1]
    m, wgt = _bm(n)
    A = sfft.rfft(np.asarray(a, np.float64), axis=-1, workers=1)
    B = sfft.rfft(np.asarray(b, np.float64), axis=-1, workers=1)
    ab = ((A * np.conj(B)).real * wgt).reshape(-1, n // 2 + 1).sum(axis=0)
    bb = ((np.abs(B) ** 2) * wgt).reshape(-1, n // 2 + 1).sum(axis=0)
    return np.array([ab[mk].sum() for mk in m]), np.array([bb[mk].sum() for mk in m])


# per-line least-squares basis over a block: DC, slope and the fsc carrier (fs/4 exactly)
_t = np.arange(BLK_W, dtype=np.float64)
_BASIS = np.stack([np.ones(BLK_W), (_t - _t.mean()) / BLK_W, np.cos(np.pi * _t / 2), np.sin(np.pi * _t / 2)], 1)
_PROJ = _BASIS @ np.linalg.pinv(_BASIS)


def block_residual(fld, y0, x0):
    """A 16x64 block minus each line's DC + slope + flat-colour carrier: what is left is noise
    (and texture).  Scaled for the 4 fitted degrees of freedom."""
    b = fld[y0:y0 + BLK_H, x0:x0 + BLK_W].astype(np.float64)
    return (b - b @ _PROJ.T) * np.sqrt(BLK_W / (BLK_W - 4))


def blocks_ms(fld, blocks):
    if not len(blocks):
        return np.full(len(BANDS), np.nan)
    return np.mean([band_ms(block_residual(fld, y, x)) for y, x in blocks], axis=0)


def block_grid(step_y=16, step_x=48):
    ys = range(RA.start, RA.stop - BLK_H, step_y)
    xs = range(CA.start, CA.stop - BLK_W, step_x)
    return [(y, x) for y in ys for x in xs]


def chroma_analytic(x):
    """Complex chroma (positive-frequency 3.5-5.5 MHz band) of rows x n."""
    n = x.shape[-1]
    f = np.fft.fftfreq(n, 1.0 / FS) / 1e6
    F = sfft.fft(np.asarray(x, np.float64), axis=-1, workers=1)
    F[..., ~((f >= 3.5) & (f < 5.5))] = 0
    return sfft.ifft(F, axis=-1, workers=1) * 2


def luma_lp(x, mhz=3.0):
    n = x.shape[-1]
    f = np.fft.rfftfreq(n, 1.0 / FS) / 1e6
    F = sfft.rfft(np.asarray(x, np.float64), axis=-1, workers=1)
    F[..., f > mhz] = 0
    return sfft.irfft(F, n, axis=-1, workers=1)


def frac_shift(x, dx):
    """Shift rows of x right by dx samples (Fourier)."""
    n = x.shape[-1]
    k = np.fft.rfftfreq(n)
    return sfft.irfft(sfft.rfft(np.asarray(x, np.float64), axis=-1) * np.exp(-2j * np.pi * k * dx), n, axis=-1)


def measure_dx(x, ref, mhz=3.0):
    """Horizontal shift of x relative to ref (x ~ ref shifted right by dx), phase-slope fit of the
    0.2-mhz MHz cross spectrum averaged over the rows."""
    n = x.shape[-1]
    f = np.fft.rfftfreq(n, 1.0 / FS) / 1e6
    X = sfft.rfft(np.asarray(x, np.float64), axis=-1)
    R = sfft.rfft(np.asarray(ref, np.float64), axis=-1)
    c = (X * np.conj(R)).sum(axis=0)
    # coarse integer search first: phase wraps beyond ~1.5 samples at 3 MHz
    cc = sfft.irfft(c, n)
    i0 = int(np.argmax(cc))
    d0 = i0 if i0 < n // 2 else i0 - n
    c = c * np.exp(2j * np.pi * np.fft.rfftfreq(n) * d0)
    sel = (f > 0.2) & (f < mhz)
    k = 2 * np.pi * np.fft.rfftfreq(n)[sel]
    ph = np.angle(c[sel])
    w = np.abs(c[sel])
    d1 = -np.sum(w * k * ph) / np.sum(w * k * k)
    return d0 + d1


# ---- picture keys, as stack.py: CAV picture number, or CLV absolute frame from the time codes
def cav_picture(vbi):
    for x in vbi or []:
        h = "%06x" % x
        if h[0] == "f" and all(c in "0123456789" for c in h[1:]):
            return int(h[1:])
    return None


def _bcd2(x):
    hi, lo = (x >> 4) & 0xF, x & 0xF
    return None if hi > 9 or lo > 9 else hi * 10 + lo


def clv_frame(vbi):
    sec = pic = hr = mn = None
    for v in vbi:
        if v is None:
            continue
        if (v & 0xF0F000) == 0x80E000:
            x1, su, p = (v >> 16) & 0xF, (v >> 8) & 0xF, _bcd2(v & 0xFF)
            if x1 >= 0xA and su <= 9 and p is not None:
                sec, pic = 10 * (x1 - 0xA) + su, p
        elif (v & 0xF0FF00) == 0xF0DD00:
            h, m = (v >> 16) & 0xF, _bcd2(v & 0xFF)
            if h <= 9 and m is not None:
                hr, mn = h, m
    if sec is not None and hr is not None:
        return ((hr * 60 + mn) * 60 + sec) * 25 + pic
    return None


def frame_keys(fields, clv=False):
    """Picture key of every frame (field pairs 2j, 2j+1) of a .tbc.json field list, None if unreadable."""
    f = cav_picture if not clv else clv_frame
    out = []
    for j in range(len(fields) // 2):
        v = (fields[2 * j].get("vbi") or {}).get("vbiData", []) + (fields[2 * j + 1].get("vbi") or {}).get("vbiData", [])
        out.append(f(v))
    return out
