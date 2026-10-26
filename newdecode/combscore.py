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
