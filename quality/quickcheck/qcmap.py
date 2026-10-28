# SPDX-License-Identifier: GPL-3.0-or-later
"""Map detail: thin-line / text edge sharpness and text-area detail energy.

Edge sharpness: the 10-90 % rise width (samples, 4fsc = 56 ns) of strong horizontal luma edges - map
lines, lettering, building outlines.  Edge positions are fixed per picture (found on the production
frame); on each frame measured the edge is re-located within +-2 samples, followed both ways while
monotone (at most 8 samples), and the 10 % / 90 % crossings interpolated.  Composite: on the
[1,2,1]/4 luma (cancels the fs/4 carrier exactly, the same filter for every candidate).
Text-area detail: 2-5 MHz energy of text / line blocks less that of flat paper blocks of the same
picture (blocks chosen on the other field / row parity of the production frame).
"""
import numpy as np

import qcmetrics as qm

EDGE_MIN = 8000           # step height (16-bit) for an edge to count: map ink on paper is 15-25k
MAX_EDGES = 400
BAND = (2.0, 5.0)


def _lum(rows):
    r = np.asarray(rows, np.float64)
    out = r.copy()
    out[..., 1:-1] = (r[..., :-2] + 2 * r[..., 1:-1] + r[..., 2:]) * 0.25
    return out


def _band_ms(b):
    n = b.shape[-1]
    f = np.fft.rfftfreq(n, 1.0 / qm.FS) / 1e6
    wgt = np.full(n // 2 + 1, 2.0)
    wgt[0] = 1
    if n % 2 == 0:
        wgt[-1] = 1
    P = (np.abs(np.fft.rfft(b, axis=-1)) ** 2 * wgt).reshape(-1, n // 2 + 1).sum(axis=0)
    return float(P[(f >= BAND[0]) & (f < BAND[1])].sum() / (b.size * n))


def find_edges(img, rows, x0, x1, rng, nmax=MAX_EDGES):
    """[(row, x)] of strong edges (gradient peaks with a > EDGE_MIN step within +-5) in a luma image."""
    from numpy.lib.stride_tricks import sliding_window_view as win
    rows = list(rows)
    p = img[rows][:, x0:x1]
    g = np.zeros_like(p)
    g[:, 1:-1] = p[:, 2:] - p[:, :-2]
    a = np.abs(g)
    pk = np.zeros_like(a, bool)
    pk[:, 1:-1] = (a[:, 1:-1] >= a[:, :-2]) & (a[:, 1:-1] > a[:, 2:])
    w = win(p, 11, axis=1)
    span = np.zeros_like(p)
    span[:, 5:-5] = w.max(axis=-1) - w.min(axis=-1)
    pk &= span > EDGE_MIN
    pk[:, :10] = pk[:, -10:] = False
    ys, xs = np.nonzero(pk)
    cand = [(rows[y], x0 + x) for y, x in zip(ys, xs)]
    if len(cand) > nmax:
        cand = [cand[i] for i in sorted(rng.choice(len(cand), nmax, replace=False))]
    return [(int(y), int(x)) for y, x in cand]


def rise_width(p, x0):
    """10-90 % width of the edge near x0 in profile p, or None."""
    g = np.zeros_like(p)
    g[1:-1] = p[2:] - p[:-2]
    xc = x0 - 2 + int(np.argmax(np.abs(g[x0 - 2:x0 + 3])))
    s = 1.0 if g[xc] > 0 else -1.0
    a = xc
    while a > xc - 8 and s * (p[a] - p[a - 1]) > 0:
        a -= 1
    b = xc
    while b < xc + 8 and s * (p[b + 1] - p[b]) > 0:
        b += 1
    h = p[b] - p[a]
    if abs(h) < EDGE_MIN / 2:
        return None
    seg = s * (p[a:b + 1] - p[a]) / abs(h)          # 0 -> 1, monotone
    xs = np.arange(a, b + 1, dtype=np.float64)
    return float(np.interp(0.9, seg, xs) - np.interp(0.1, seg, xs))


# ---- noise-robust edge width: profiles averaged along the edge, erf fit, plateaux at fixed +-6 px
STEP_ROWS = 8        # rows either side averaged along the edge (17 rows)
STEP_PLATEAU = 6     # plateau samples taken at -8..-5 and +5..+8 from the centre
ERF_K = 1.8124       # 10-90 % width of 0.5 (1 + erf(x / s)) is 1.8124 s


def _centre(p, x0):
    """Sub-pixel position of the steepest point of row profile p near x0 (parabola on |gradient|)."""
    g = np.zeros_like(p)
    g[1:-1] = p[2:] - p[:-2]
    if x0 - 3 < 1 or x0 + 4 >= len(p) - 1:
        return None, 0.0
    xc = x0 - 3 + int(np.argmax(np.abs(g[x0 - 3:x0 + 4])))
    a, b, c = np.abs(g[xc - 1:xc + 2])
    den = a - 2 * b + c
    return xc + (0.5 * (a - c) / den if den < 0 else 0.0), float(g[xc])


def step_geometry(img, y, x, rows=STEP_ROWS):
    """On the REFERENCE: the edge's sub-pixel centre on each row y-rows..y+rows (None where the row
    does not carry the same edge) and its direction; None unless >= 2/3 of the rows carry it."""
    _, g0 = _centre(img[y], x)
    cs = []
    for yy in range(y - rows, y + rows + 1):
        c, g = (None, 0.0) if not 0 <= yy < img.shape[0] else _centre(img[yy], x)
        cs.append(None if c is None or g * g0 <= 0 or abs(g) < 0.3 * abs(g0) else round(float(c), 3))
    if sum(c is not None for c in cs) < (2 * rows + 1) * 2 // 3:
        return None
    return {"y": int(y), "sign": 1 if g0 > 0 else -1, "c": cs}


XS = np.arange(-10, 11, dtype=np.float64)


def _crossing(p, c, xs=None):
    """Sub-pixel position near c where row profile p (oriented low -> high) crosses half-way between
    its plateaux at c-8..c-5 and c+5..c+8; None if not found."""
    base = np.arange(len(p), dtype=np.float64)
    q = np.interp(c + XS, base, p)
    lo, hi = q[(XS >= -8) & (XS <= -5)].mean(), q[(XS >= 5) & (XS <= 8)].mean()
    if hi - lo <= 0:
        return None
    mid = 0.5 * (lo + hi)
    w = np.nonzero((XS[:-1] >= -4) & (XS[:-1] <= 3) & (q[:-1] < mid) & (q[1:] >= mid))[0]
    if len(w) == 0:
        return None
    i = w[np.argmin(np.abs(XS[w]))]                 # the crossing nearest the reference centre
    return c + XS[i] + (mid - q[i]) / (q[i + 1] - q[i])


def _norm_profile(img, geo):
    """The edge's profile on img, oriented low -> high, each row aligned on ITS OWN half-way crossing
    (found near the reference centre), averaged along the edge, plateaux at fixed -8..-5 / +5..+8 ->
    0..1.  Aligning each row on itself makes the measure blind to per-line timing jitter (single disc
    frames carry 0.5-0.9 px of it; a held-still average does not) and to the reference's own jitter.
    None if the step vanished."""
    base = np.arange(img.shape[1], dtype=np.float64)
    acc, n = np.zeros(len(XS)), 0
    # the row span is the one the edge was found with (stored with it), not today's STEP_ROWS: a spec
    # made with 13 rows measured with 17-row offsets would read the wrong rows (north2 52200: 3.1 -> 7.5)
    rows = (len(geo["c"]) - 1) // 2
    for i, c in enumerate(geo["c"]):
        if c is None:
            continue
        row = geo["sign"] * img[geo["y"] - rows + i].astype(np.float64)
        cc = _crossing(row, c)
        if cc is None or abs(cc - c) > 3:
            continue
        acc += np.interp(cc + XS, base, row)
        n += 1
    if n < 3:
        return None
    p = acc / n
    lo, hi = p[(XS >= -8) & (XS <= -5)].mean(), p[(XS >= 5) & (XS <= 8)].mean()
    if hi - lo < EDGE_MIN / 4:
        return None
    return (p - lo) / (hi - lo)


def erf_width(n):
    """10-90 % width from a least-squares fit of 0.5 (1 + erf((x - c) / s)) to a 0..1 step over -5..+5."""
    from scipy.optimize import least_squares
    from scipy.special import erf
    m = (XS >= -5) & (XS <= 5)
    f = lambda q: 0.5 * (1 + erf((XS[m] - q[0]) / max(q[1], 0.05))) - n[m]
    r = least_squares(f, [0.0, 1.5], bounds=([-2, 0.1], [2, 8]))
    return float(ERF_K * r.x[1]), float(np.sqrt(np.mean(r.fun ** 2)))


def find_steps(img, rows, x0, x1, rng, nmax=MAX_EDGES):
    """Edges for the robust measure, on the reference: strong gradient peaks that run along >= 2/3 of
    13 rows and whose averaged profile is a clean single step (erf fit residual < 0.1)."""
    out = []
    for y, x in find_edges(img, rows, x0 + 12, x1 - 12, rng, 4000):
        if y - STEP_ROWS < rows[0] or y + STEP_ROWS > rows[-1]:
            continue
        geo = step_geometry(img, y, x)
        if geo is None:
            continue
        n = _norm_profile(img, geo)
        if n is None or erf_width(n)[1] > 0.10:
            continue
        out.append(geo)
        if len(out) >= nmax:
            break
    return out


def robust_rise(img, steps):
    """One width per picture: the mean of all its steps' normalised along-edge profiles, erf-fitted.
    Thousands of samples go into the one profile, so noise cannot shorten or lengthen the edge."""
    ps = [p for p in (_norm_profile(img, g) for g in steps) if p is not None]
    if len(ps) < 3:
        return None, len(ps)
    return erf_width(np.mean(ps, axis=0))[0], len(ps)


def _blocks(img_other, ys, xs, bh, bw, lvl_img):
    """Per block (y, x): 2-5 MHz energy on the other field / parity, and level."""
    out = []
    for y in ys:
        for x in xs:
            b = img_other(y, x)
            b = b - b.mean(axis=1, keepdims=True)
            out.append((y, x, _band_ms(b), float(lvl_img[y:y + bh, x:x + bw].mean())))
    return out


def select(ref, Y, seed=0):
    """Fixed regions for one map picture: ref = production composite frame (2,H,W), Y = its
    transform2d luma (576,928)."""
    rng = np.random.default_rng(seed)
    spec = {"comp": {"edges": [], "text": [], "paper": []}, "dec": {"edges": [], "text": [], "paper": []}}
    for f in (0, 1):
        lum = _lum(ref[f])
        e = find_edges(lum, range(qm.RA.start, qm.RA.stop), qm.CA.start, qm.CA.stop, rng)
        spec["comp"]["edges"] += [[f, y, x] for y, x in e[:MAX_EDGES // 2]]
        spec["comp"].setdefault("steps", [])
        spec["comp"]["steps"] += [dict(g, f=f) for g in find_steps(lum, list(range(qm.RA.start, qm.RA.stop)),
                                                                   qm.CA.start, qm.CA.stop, rng, 200)]
        o = ref[1 - f].astype(np.float64)
        bl = []
        for y, x in qm.block_grid():
            r = qm.block_residual(o, y, x)
            bl.append((y, x, _band_ms(r), float(o[y:y + 16, x:x + 64].mean())))
        en = np.array([b[2] for b in bl])
        lv = np.array([b[3] for b in bl])
        ti = [i for i in np.argsort(-en) if lv[i] > 25000][:16]
        pi = [i for i in np.argsort(en) if lv[i] > 30000][:16]
        spec["comp"]["text"] += [[f, bl[i][0], bl[i][1]] for i in ti]
        spec["comp"]["paper"] += [[f, bl[i][0], bl[i][1]] for i in pi]
    Yf = Y.astype(np.float64)
    e = find_edges(Yf, range(20, Y.shape[0] - 20), 8, Y.shape[1] - 8, rng)
    spec["dec"]["edges"] = [[y, x] for y, x in e]
    spec["dec"]["steps"] = find_steps(Yf, list(range(20, Y.shape[0] - 20)), 8, Y.shape[1] - 8, rng, 400)
    for par in (0, 1):
        ys, xs = range(16, Y.shape[0] - 48, 32), range(16, Y.shape[1] - 80, 48)
        bl = _blocks(lambda y, x: Yf[y + 1 - par:y + 32:2, x:x + 64], ys, xs, 32, 64, Yf)
        en = np.array([b[2] for b in bl])
        lv = np.array([b[3] for b in bl])
        ti = [i for i in np.argsort(-en) if lv[i] > 20000][:16]
        pi = [i for i in np.argsort(en) if lv[i] > 30000][:16]
        spec["dec"]["text"] += [[par, bl[i][0], bl[i][1]] for i in ti]
        spec["dec"]["paper"] += [[par, bl[i][0], bl[i][1]] for i in pi]
    return spec


def measure_comp(fr, spec):
    """fr: candidate composite frame (2,H,W)."""
    s = spec["comp"]
    lum = [_lum(fr[0]), _lum(fr[1])]
    w = [rise_width(lum[f][y], x) for f, y, x in s["edges"]]
    w = [x for x in w if x is not None]
    t = [_band_ms(qm.block_residual(fr[f].astype(np.float64), y, x)) for f, y, x in s["text"]]
    p = [_band_ms(qm.block_residual(fr[f].astype(np.float64), y, x)) for f, y, x in s["paper"]]
    out = {"rise_walk": float(np.mean(w)) if w else None, "n_walk": len(w)}
    if s.get("steps"):
        # both fields' steps into one profile: lum[f] per step
        ps = [p for p in (_norm_profile(lum[g["f"]], g) for g in s["steps"]) if p is not None]
        out["rise"], out["n_edges"] = ((erf_width(np.mean(ps, axis=0))[0], len(ps)) if len(ps) >= 3
                                       else (None, len(ps)))
    else:
        out["rise"], out["n_edges"] = out["rise_walk"], len(w)
    return {**out,
            "text_ms": float(np.mean(t)) if t else None, "paper_ms": float(np.mean(p)) if p else None}


def measure_dec(Y, spec):
    """Y: candidate decoded luma (576,928)."""
    s = spec["dec"]
    Yf = Y.astype(np.float64)
    w = [rise_width(Yf[y], x) for y, x in s["edges"]]
    w = [x for x in w if x is not None]
    out = {"rise_walk": float(np.mean(w)) if w else None, "n_walk": len(w)}
    if s.get("steps"):
        out["rise"], out["n_edges"] = robust_rise(Yf, s["steps"])
    else:
        out["rise"], out["n_edges"] = out["rise_walk"], len(w)

    def bms(blocks):
        v = []
        for par, y, x in blocks:
            b = Yf[y + par:y + 32:2, x:x + 64]
            v.append(_band_ms(b - b.mean(axis=1, keepdims=True)))
        return float(np.mean(v)) if v else None
    return {**out, "text_ms": bms(s["text"]), "paper_ms": bms(s["paper"])}


def summarise(ms):
    """[measure_* dicts] -> mean rise width, text-area net 2-5 MHz rms, paper 2-5 MHz rms."""
    ms = [m for m in ms if m and m.get("rise") is not None]
    if not ms:
        return {}
    w = np.array([m["rise"] for m in ms])
    n = np.array([m["n_edges"] for m in ms], np.float64)
    t = np.array([m["text_ms"] for m in ms if m["text_ms"] is not None and m["paper_ms"] is not None])
    p = np.array([m["paper_ms"] for m in ms if m["text_ms"] is not None and m["paper_ms"] is not None])
    return {"rise": float((w * n).sum() / n.sum()), "edges": int(n.sum()), "frames": len(ms),
            "text": float(np.sqrt(max((t - p).mean(), 0))) if len(t) else None,
            "paper": float(np.sqrt(p.mean())) if len(p) else None}
