# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-capture error covariance on the panel's held runs, independent of any stacker.

  measure_masters.py <panel dir> [--frames-per-run N]

Each capture's field is registered to the run's temporal mean (the production stack averaged over
frames of the run that the panel does not stack): horizontal shift (phase-slope fit, 0.2-3 MHz),
chroma phase, gain and offset; what is left is that capture's error (disc noise of its master +
its own player noise).  Lines with a dropout on any capture are left out.  The per-band covariance
C[b][i][j] of those residuals (minus the run mean's own noise) gives the error variance of any
grouping / weighting of the captures, so quickcheck can judge the stacker's master blend weights
against what the masters' noise really is.  Written into panel.json as "master_cov".
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcmetrics as qm          # noqa: E402


def cap_index(base, clv=False):
    m = json.load(open(base + ".tbc.json"))
    f = m["fields"]
    idx = {}
    for j, k in enumerate(qm.frame_keys(f, clv)):
        if k is not None and k not in idx:
            idx[k] = j
    return idx, f


def dropout_rows(fj):
    do = fj.get("dropOuts") or {}
    return {int(l) - 1 for l in do.get("fieldLine", [])}     # fieldLine is 1-based


def register(x, ref, fit=None):
    """x (rows x n, float) aligned to ref: shift, chroma complex gain (phase + saturation), luma
    gain/offset, fitted on rows `fit` (default all: e.g. the rows without a dropout on this capture)
    and applied to every row; -> (residual, dx, chroma rotation deg, gain)."""
    fit = np.arange(len(x)) if fit is None else np.asarray(fit)
    if len(fit) < 20 or ref[fit].std() < 1500:
        # no picture to register or scale against (black / flat), or too few clean rows: offset only
        o = (x[fit] - ref[fit]).mean() if len(fit) else (x - ref).mean()
        return x - o - ref, 0.0, 0.0, 1.0
    dx = qm.measure_dx(qm.luma_lp(x[fit]), qm.luma_lp(ref[fit]))
    xs = qm.frac_shift(x, -dx)
    z = qm.chroma_analytic(xs)
    zr = qm.chroma_analytic(ref)
    c = (zr[fit] * np.conj(z[fit])).sum() / max((np.abs(z[fit]) ** 2).sum(), 1e-9)
    xs = xs - z.real + (z * c).real
    A = np.stack([ref[fit].ravel(), np.ones(ref[fit].size)], 1)
    g, o = np.linalg.lstsq(A, xs[fit].ravel(), rcond=None)[0]
    return (xs - o) / g - ref, dx, -np.degrees(np.angle(c)), g


def common_part(C, N=None, valid=None):
    """Error every capture shares (the source's own per-frame noise: no stacker can remove it and no
    weighting changes it): a low percentile (20 %) of the covariances between distinct captures,
    which on a side with several masters is the cross-master level.  Masters: correlation of the
    rest > 0.3 (2-3.5 MHz).  Only measured pairs count."""
    n = C.shape[1]
    valid = list(range(n)) if valid is None else valid
    pairs = [(i, j) for i in valid for j in valid if i < j and (N is None or N[i, j] > 0)
             and np.isfinite(C[1, i, j])]
    if not pairs:
        return [0.0] * C.shape[0], list(range(n))
    common = [float(np.percentile([C[b, i, j] for i, j in pairs], 20)) for b in range(C.shape[0])]
    Cr = C[1] - common[1]
    d = np.sqrt(np.maximum(np.diag(Cr), 1e-9))
    R = Cr / np.outer(d, d)
    lab = list(range(n))
    for i, j in pairs:
        if R[i, j] > 0.3:
            a, b = lab[i], lab[j]
            lab = [a if x == b else x for x in lab]
    return common, lab


def base_name(n):
    """One capture recorded in parts (NationalA ds1a / ds1b) counts once for master noise."""
    m = re.match(r"^(.*ds\d+)[a-z]$", n)
    return m.group(1) if m else n


MIN_ROWS = 40        # a pair needs this many clean rows in a field pair to count
MIN_PAIRS = 4        # ... and this many field pairs for its covariance to be used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--frames-per-run", type=int, default=4)
    a = ap.parse_args()
    pj = os.path.join(a.panel, "panel.json")
    panel = json.load(open(pj))
    parts = panel["caps"]
    caps = {}
    for n in parts:
        b = os.path.join(a.panel, "caps", n)
        idx, f = cap_index(b, panel.get("clv", False))
        caps[n] = (idx, f, np.memmap(b + ".tbc", "<u2", "r").reshape(-1, qm.H, qm.W))
    names = [n for n in dict.fromkeys(base_name(p) for p in parts)]

    def frame(n, k):
        """(fields meta, tbc, frame index) of picture k on capture n (any of its parts)."""
        for p in parts:
            if base_name(p) == n and k in caps[p][0]:
                return caps[p][1], caps[p][2], caps[p][0][k]
        return None
    nb = len(qm.BANDS)
    S = np.zeros((nb, len(names), len(names)))
    F = np.zeros((nb, len(names), len(names)))
    N = np.zeros((len(names), len(names)))
    shifts = {n: [] for n in names}
    all_rows = np.arange(qm.RA.start, qm.RA.stop)
    runs_used = []
    per_run = {}
    for r in panel["runs"]:
        ks = [k for k in r["stacked"] if str(r["phase"][str(k)]) in r["floor"]][:a.frames_per_run]
        if len(ks) < 2:
            continue
        runs_used.append(r["name"])
        S0, F0, N0 = S.copy(), F.copy(), N.copy()
        for fld in (0, 1):
            present = [n for n in names if all(frame(n, k) for k in ks)]
            E = {n: [] for n in present}
            G = {n: [] for n in present}          # clean rows (no dropout on this capture), per frame
            for k in ks:
                P = str(r["phase"][str(k)])
                ref = np.load(os.path.join(a.panel, "ref", f"runmean_{r['r0']}_{P}.npy"))[fld][qm.RA, qm.CA]
                ref = ref.astype(np.float64)
                for n in present:
                    meta, tbc, j = frame(n, k)
                    bad = dropout_rows(meta[2 * j + fld])
                    good = np.array([y not in bad for y in all_rows])
                    x = tbc[2 * j + fld][qm.RA, qm.CA].astype(np.float64)
                    e, dx, rot, g = register(x, ref, np.nonzero(good)[0])
                    if g == 1.0 and dx == 0.0 and ref.std() >= 1500:
                        good[:] = False                 # could not be registered: not used
                    else:
                        shifts[n].append((dx, rot, g))
                    E[n].append(e)
                    G[n].append(good)
            # consecutive frames' residual differences: what is fixed from frame to frame (the
            # reference's own bias, a master's systematic response) cancels; the random error stays.
            # Each pair uses the rows clean on both captures in both frames.
            for t in range(len(ks) - 1):
                P1, P2 = str(r["phase"][str(ks[t])]), str(r["phase"][str(ks[t + 1])])
                fl = (np.array(r["floor"][P1]) + np.array(r["floor"][P2])) / 2
                D = {n: (E[n][t] - E[n][t + 1]) / np.sqrt(2) for n in present}
                g2 = {n: G[n][t] & G[n][t + 1] for n in present}
                for i, ni in enumerate(names):
                    for j2, nj in enumerate(names):
                        if j2 >= i and ni in D and nj in D:
                            rows = g2[ni] & g2[nj]
                            if rows.sum() < MIN_ROWS:
                                continue
                            di, dj = D[ni][rows], D[nj][rows]
                            ab, _ = qm.band_cross(di, dj)
                            S[:, i, j2] += ab / (di.size * di.shape[-1])
                            F[:, i, j2] += fl
                            N[i, j2] += 1
        per_run[f"{r['name']}@{r['r0']}"] = {"S": (S - S0).tolist(), "F": (F - F0).tolist(), "N": (N - N0).tolist()}
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(N >= MIN_PAIRS, (S - F) / np.maximum(N, 1), np.nan)
    for i in range(len(names)):
        for j in range(i):
            C[:, i, j] = C[:, j, i]
            N[i, j] = N[j, i]
    valid = [i for i in range(len(names)) if N[i, i] >= MIN_PAIRS and np.all(C[:, i, i] > 0)]
    common, lab = common_part(C, N, valid)
    panel["master_cov"] = {"names": names, "C": np.where(np.isfinite(C), C, None).tolist(),
                           "N": N.tolist(), "valid": [names[i] for i in valid], "parts": parts, "runs_used": runs_used, "per_run": per_run,
                           "bands": qm.BAND_NAMES, "common": common, "clusters": lab,
                           "shifts": {n: np.median(v, axis=0).tolist() for n, v in shifts.items() if v}}
    json.dump(panel, open(pj + ".tmp", "w"), indent=1)
    os.replace(pj + ".tmp", pj)
    print("capture   dx     rot    gain   rms per band (own error)")
    for i, n in enumerate(names):
        s = panel["master_cov"]["shifts"].get(n, [0, 0, 0])
        print(f"{n:22} {s[0]:+.2f} {s[1]:+6.1f} {s[2]:.3f}  " + " ".join(
            f"{np.sqrt(max(C[b, i, i], 0)) if np.isfinite(C[b, i, i]) else float('nan'):7.1f}" for b in range(nb))
            + f"   pairs {int(N[i, i])}" + ("" if i in valid else "  (insufficient)"))
    print("common (shared by all captures) per band:", [round(x) for x in common], " masters:", lab)
    print("correlation (2-3.5 MHz):")
    C1 = np.nan_to_num(C[1])
    d = np.sqrt(np.maximum(np.diag(C1), 1e-9))
    R = C1 / np.outer(d, d)
    print("       " + " ".join(f"{n.split('_')[-1]:>5}" for n in names))
    for i, n in enumerate(names):
        print(f"{n.split('_')[-1]:>6} " + " ".join(f"{R[i, j]:5.2f}" for j in range(len(names))))


if __name__ == "__main__":
    main()
