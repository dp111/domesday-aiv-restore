# SPDX-License-Identifier: GPL-3.0-or-later
"""Inter-field motion score for a woven frame (Y, 16-bit).

Each line is predicted two ways: from the OTHER field's lines above and
below (half the distance, so on a still they predict it better) and from its
OWN field's lines (a full frame-line pair away).  Motion between the fields
spoils the first prediction but not the second.  Score = inter / intra
prediction error, summed over edge pixels (flat areas are only noise), on
luma low-passed well below the subcarrier so colour residue cannot count as motion.
~<1: a single instant (still / film) -> weave;  >>1: interlaced motion."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _errors(Y):
    # Gaussian sigma 2 samples: -43 dB at 4.43 MHz.  A 5-sample box (the first
    # version) passes enough subcarrier residue (herringbone) that colour bars
    # read as inter-field motion: the residue's phase differs between fields.
    Y = gaussian_filter1d(Y.astype(np.float32), 2.0, axis=1)
    c = Y[2:-2]
    inter = np.abs(c - 0.5 * (Y[1:-3] + Y[3:-1]))
    intra = np.abs(c - 0.5 * (Y[:-4] + Y[4:]))
    # Inter-field error only counts as motion when it exceeds the noise: on
    # content with no vertical detail (colour bars) both predictions are
    # perfect bar noise, and inter/intra alone reads ~1 - "motion".
    noise = 1.4826 * np.median(intra)
    intra = np.maximum(intra, 2.0 * noise)
    g = np.abs(Y[3:-1] - Y[1:-3]) + np.abs(np.gradient(c, axis=1))
    return inter, intra, g


def comb_score(Y, block=None):
    inter, intra, g = _errors(Y)
    edge = g > np.percentile(g, 80)
    if block is None:
        return float(inter[edge].sum() / (intra[edge].sum() + 1e-6))
    by, bx = block
    h, w = (inter.shape[0] // by) * by, (inter.shape[1] // bx) * bx
    m = edge[:h, :w]
    a = (inter[:h, :w] * m).reshape(h // by, by, w // bx, bx).sum((1, 3))
    b = (intra[:h, :w] * m).reshape(h // by, by, w // bx, bx).sum((1, 3)) + 1e-6
    n = m.reshape(h // by, by, w // bx, bx).sum((1, 3))
    return np.where(n > 8, a / b, 0.0)


def match_score(Y, Yn):
    """Frame-to-frame match of two woven frames (Y, 16-bit): luma difference on the edge pixels of Y over the
    same pixels' same-field prediction error (as comb_score, noise-floored).  A repeated still (held run) differs
    only by noise: ~<0.3; any motion, even sub-pixel, shows at the edges: >>1.  (2026-10-27, herring classifier)"""
    a = gaussian_filter1d(Y.astype(np.float32), 2.0, axis=1)
    b = gaussian_filter1d(Yn.astype(np.float32), 2.0, axis=1)
    _, intra, g = _errors(Y)
    edge = g > np.percentile(g, 80)
    d = np.abs(a - b)[2:-2]
    return float(d[edge].sum() / (intra[edge].sum() + 1e-6))


# ---- still / interlaced decision (2026-10-27, herring classifier) ------------------------------------------------
# comb_score alone puts static saturated colour (EBU bars 0.59-0.63, menus / satellite stills 0.58-0.67) next to the
# 0.58 threshold: the PAL field-phase residue of strong flat colour reads as inter-field motion, and anything that
# lowers the noise (a better stack, the herringbone canceller) tips them to 'interlaced', which loses the held-run
# averaging (South bars: grain x3).  Real interlaced motion moves between FRAMES as well: match_score to the
# same-scene neighbour is >= 0.7 (South 61-69, 168-194, 44997-45013, 49229-49243, 52954-52969) where repeated
# stills give 0.05-0.20 (bars 0.48: few edges, mostly noise).  Rule (frame i, c = comb_score, m = its smallest
# match_score to a same-scene neighbour, m = inf without one):
#   strong  = c > C_HI or (c > C_T and m >= M_MOVE)
#   interlaced = strong, or c > C_T and (m >= M_HELD or a strong frame lies within NEAR frames in the same scene)
# so the decelerating tail of a motion run (c 0.58-0.64, m 0.2-0.4, next to strong frames) stays interlaced and a
# frame with no same-scene neighbour keeps the old c > C_T rule.
import os as _os
C_T = float(_os.environ.get('PALDEC_COMB_T', '0.58'))
C_HI = float(_os.environ.get('PALDEC_COMB_HI', '0.70'))
M_MOVE, M_HELD = 0.7, 0.6
NEAR = 4


def comb_lowsat(X, sat=12.0):
    """comb_score over the edge pixels of LOW-saturation picture only (X (3, H, W) 16-bit YCbCr; |C| < sat 8-bit
    levels, 5 px max-filtered): the PAL field-phase residue of strong colour cannot inflate it.  Bars 0.45-0.54, a
    text page 0.55-0.60; computer graphics whose fields really differ (graph paper, menus) 0.62-0.83.  NaN when
    the picture has fewer than 500 such edge pixels."""
    from scipy.ndimage import maximum_filter
    inter, intra, g = _errors(X[0])
    edge = g > np.percentile(g, 80)
    C = maximum_filter(np.hypot(X[1] - 32768., X[2] - 32768.) / 256, size=5)[2:-2]
    lo = edge & (C < sat)
    return float(inter[lo].sum() / (intra[lo].sum() + 1e-6)) if lo.sum() >= 500 else float('nan')


C_LOWSAT = float(_os.environ.get('PALDEC_COMB_LOWSAT', '0.58'))


def classify_interlaced(c, m, link, cl=None):
    """c, m: per-frame comb_score / match_score (np arrays); link[i]: frame i is the same scene as i-1; cl:
    comb_lowsat per frame (optional; NaN = no low-saturation edges).  -> bool array: interlaced."""
    c = np.asarray(c, float); m = np.asarray(m, float); n = len(c)
    strong = (c > C_HI) | ((c > C_T) & (m >= M_MOVE))
    scene = np.cumsum([0] + [0 if link[i] else 1 for i in range(1, n)])
    near = np.zeros(n, bool)
    for i in np.flatnonzero(strong):
        lo, hi = max(0, i - NEAR), min(n, i + NEAR + 1)
        near[lo:hi] |= scene[lo:hi] == scene[i]
    inter = strong | ((c > C_T) & ((m >= M_HELD) | near))
    # 2026-10-28: the override to 'still' only where the combing is explained by colour.  Static graphics whose two
    # fields really differ (North graph paper 2237: c 0.63, m 0.16, low-saturation edges 0.81) zipper when woven,
    # and saturated menus solved as film keep a subcarrier hatch; both stay interlaced as before.
    if cl is not None:
        cl = np.asarray(cl, float).copy()
        for sc in np.unique(scene):          # one decision per same-scene run (its median), no flicker between classes
            sel = scene == sc
            if np.isfinite(cl[sel]).any():
                cl[sel] = np.nanmedian(cl[sel])
        keep = (c > C_T) & ~inter & ~(cl <= C_LOWSAT)       # NaN (no low-sat edges, e.g. bars) -> may override
        keep &= ~np.isnan(cl)
        inter = inter | keep
    return inter
