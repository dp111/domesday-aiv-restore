"""Exact PAL forward model for ld-decode 4fsc TBC fields.

The TBC is subcarrier-locked: with k the sample index and l the field line,
the subcarrier phase advances 90 deg per sample and (90 + D) deg per line,
D = -0.576 deg being the 25 Hz offset (fsc = 283.75 fH + 25 Hz) that a
whole-sample line length leaves behind.  Each line's burst therefore sits at

    beta_l = beta0 + (90 + D) * l + (0 or 90 deg by the PAL burst swing)

with beta0 one value per field (measured to +-2 deg, fitted over all lines),
and the swing gives the V-switch sign of the line.  A field's composite is

    comp = A_Y * Y + B + G_U * U * cos(pi/2 k - phi_l)
                       + G_V * V * cos(pi/2 k - phi_l - s_l * pi/2)

with phi_l = beta_l + C_U, all in the units of ld-chroma-decoder's
yuv444p16 output (Y limited range 16-bit, U/V centred on 32768).
C_U, the gains and the luma scale were calibrated against ld-chroma-decoder
on Community South (calibrate()), giving global R^2 0.985.
"""
import numpy as np

W, H = 1135, 313
D_DEG = -0.576
BK = np.arange(100, 136)          # burst samples


def burst_phase(line):
    x = line[100:136] - line[100:136].mean()
    return np.degrees(np.arctan2((x * np.sin(np.pi / 2 * BK)).sum(),
                                 (x * np.cos(np.pi / 2 * BK)).sum()))


_BC = np.cos(np.pi / 2 * BK); _BS = np.sin(np.pi / 2 * BK)


def field_burst(field, lines=range(30, 300)):
    """beta0 (deg) and which line parity carries the +90 deg burst swing."""
    ls = np.array(list(lines))
    x = np.asarray(field)[ls, 100:136].astype(np.float64)
    x = x - x.mean(1, keepdims=True)                     # all lines at once
    b = np.degrees(np.arctan2(x @ _BS, x @ _BC))
    r = b - (90 + D_DEG) * ls
    bp = [np.degrees(np.angle(np.exp(1j * np.radians(r[ls % 2 == p])).mean())) for p in (0, 1)]
    if (bp[1] - bp[0]) % 360 < 180:      # parity 1 is 90 deg ahead
        return bp[0] % 360, 1
    return bp[1] % 360, 0


import json as _json, os as _os
# PALDEC_PARAMS: alternative calibration file (A/B against another carrier calibration)
P = _json.load(open(_os.environ.get("PALDEC_PARAMS") or _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "model_params.json")))


def carriers(beta0, par, nlines=H, width=W, x0=0, line0=0):
    """Per-line carriers (nlines, width) multiplying U and V.

    The reference is the burst's mean (non-swinging) phase; the swing only
    sets the sign of V on alternate lines."""
    k = np.arange(x0, x0 + width)[None, :]
    l = np.arange(line0, line0 + nlines)[:, None]
    ref = np.radians(beta0 + (90 + D_DEG) * l)
    s = np.where((l % 2) == par, -1.0, 1.0)
    cu = P["GU"] * np.cos(np.pi / 2 * k - ref - np.radians(P["cU"]))
    cv = P["GV"] * s * np.cos(np.pi / 2 * k - ref - np.radians(P["cV"]))
    return cu, cv


def encode(Y, U, V, cu, cv):
    """Composite from Y (16-bit limited), U, V (centred on 0) with carriers."""
    return P["AY"] * Y + P["B"] + U * cu + V * cv
