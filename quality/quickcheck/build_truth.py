# SPDX-License-Identifier: GPL-3.0-or-later
"""Held-run truth from the raw captures, independent of any stacker.

  build_truth.py <panel dir> [--full-index index.json] [--frames 16] [--out-json panel_v2.json]

For each held run and PAL phase (frames k, k+4, ... share the subcarrier phase), from up to --frames
pictures of the run that the scorer does NOT score:
  1. every capture's field is registered to the run's provisional mean (shift, chroma phase), and its
     luma gain / offset and chroma gain against that mean are measured;
  2. each capture is rescaled to the MEDIAN gain / offset / chroma gain of the captures, so no single
     capture (or stacker) sets the contrast;
  3. a robust per-capture test drops gross outliers: a capture whose own random error (master_cov
     diagonal, all bands) is > 4x the median capture's (logged with its ratio).  A capture of another
     master has the same order of error as any other, so this does not drop alt masters;
  4. the frame's truth is the mean of the masters' means (captures equal within a master, masters
     equal; masters = master_cov clusters, split captures as one);
  5. the phase's truth is the mean over its frames; its own noise (frame-to-frame scatter / n) is kept
     as the floor the scorer subtracts.
Also writes truth0 = the same from the largest master only, as a sensitivity check.
Pictures not in the panel's mini captures are read from the full captures (--full-index, North).
Writes ref/truth_<r0>_<P>.npy, ref/truth0_<r0>_<P>.npy and a "truth" entry per run into
--out-json (default panel_v2.json beside panel.json: switch when no suite is reading the panel).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcmetrics as qm              # noqa: E402
from measure_masters import base_name, cap_index, dropout_rows   # noqa: E402

EDGE = 2           # pictures at each end of a run never used (fades in / out)
OUTLIER = 4.0      # gross outlier: own random error > 4x the median capture's


def fit(x, ref):
    """shift, chroma complex gain, luma gain / offset of x vs ref (rows x n)."""
    if ref.std() < 1500:
        return 0.0, 1.0 + 0j, 1.0, float((x - ref).mean())
    dx = qm.measure_dx(qm.luma_lp(x), qm.luma_lp(ref))
    xs = qm.frac_shift(x, -dx)
    z, zr = qm.chroma_analytic(xs), qm.chroma_analytic(ref)
    c = (zr * np.conj(z)).sum() / max((np.abs(z) ** 2).sum(), 1e-9)      # x chroma * c ~ ref chroma
    yl, rl = qm.luma_lp(xs), qm.luma_lp(ref)
    A = np.stack([rl.ravel(), np.ones(rl.size)], 1)
    g, o = np.linalg.lstsq(A, yl.ravel(), rcond=None)[0]                  # x luma ~ g ref + o
    return float(dx), complex(c), float(g), float(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--full-index", default=None)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    pdir = os.path.abspath(a.panel)
    panel = json.load(open(os.path.join(pdir, "panel.json")))
    clv = panel.get("clv", False)
    parts = panel["caps"]
    src = {}
    for n in parts:
        b = os.path.join(pdir, "caps", n)
        idx, f = cap_index(b, clv)
        src[n] = [(idx, f, np.memmap(b + ".tbc", "<u2", "r").reshape(-1, qm.H, qm.W))]
    if a.full_index:
        fi = json.load(open(a.full_index))["caps"]
        sys.path.insert(0, "/mnt/s/domsday/ld-decode")
        from lddecode.tbcmeta import load_decode_metadata
        for n in parts:
            if n in fi:
                c = fi[n]
                meta = load_decode_metadata(c["base"])
                src[n].append((dict(zip(c["keys"], c["fi"])), meta["fields"],
                               np.memmap(c["base"] + ".tbc", "<u2", "r").reshape(-1, qm.H, qm.W)))
    names = list(dict.fromkeys(base_name(p) for p in parts))
    mc = panel.get("master_cov") or {}
    cl = dict(zip(mc.get("names", []), mc.get("clusters", [])))
    master = {n: cl.get(n, 0) for n in names}
    big = max(set(master.values()), key=lambda m: sum(v == m for v in master.values()))

    def field(n, k, fld):
        for p in parts:
            if base_name(p) != n:
                continue
            for idx, f, tb in src[p]:
                if k in idx:
                    j = idx[k]
                    return tb[2 * j + fld][qm.RA, qm.CA].astype(np.float64), dropout_rows(f[2 * j + fld])
        return None

    # gross outliers: a capture's own random error (master_cov: residual vs the run mean, frame to frame,
    # all bands) > OUTLIER x the median capture's.  Master-independent: a capture of another master has
    # the same disc + player noise as any other, so only a capture that is itself bad stands out
    # (Countryside ds3: 26x; every other capture of the suite is within 0.5-1.5x).
    gross = {}
    if mc.get("C"):
        Cd = np.array([[np.nan if row[i] is None else row[i] for i, row in enumerate(band)] for band in mc["C"]], float)
        tot = np.nansum(Cd, axis=0)
        med = float(np.nanmedian(tot))
        gross = {n: round(float(t / med), 1) for n, t in zip(mc["names"], tot) if med > 0 and t > OUTLIER * med}
    log = [f"gross outliers (own error > {OUTLIER:g}x the median capture's): {gross or 'none'}"]
    print(log[0])
    truth = {}
    def lf(k):
        """Median over the captures of picture k's low-passed (< 1 MHz) active luma, or None."""
        fs = [field(n, k, 0) for n in names]
        fs = [f[0] for f in fs if f is not None]
        if not fs:
            return None
        return qm.luma_lp(np.median(fs, axis=0)[::4], 1.0)

    for r in panel["runs"]:
        scored = set(r["stacked"])
        info = {"frames": {}, "excluded": {}, "floor": {}, "floor0": {}, "gain_median": {},
                "transition": {}}
        # transition frames (a run fading in or out): the first and last 2 pictures of the run, and any
        # picture whose LF luma gain against the run's median picture is off by > 2 % or level by > 1 %
        cand = [k for k in range(r["r0"] + EDGE, r["r1"] - EDGE + 1) if k not in scored]
        cand = sorted(cand, key=lambda k: abs(k - np.mean(r["stacked"])))[:6 * (a.frames // 4 + 2)]
        L = {k: lf(k) for k in cand}
        L = {k: v for k, v in L.items() if v is not None}
        bad = set()
        if len(L) >= 3:
            med = np.median(list(L.values()), axis=0)
            mm = med - med.mean()
            rng_ = max(float(np.percentile(med, 99) - np.percentile(med, 1)), 1.0)
            for k, v in L.items():
                g = float(((v - v.mean()) * mm).sum() / max((mm * mm).sum(), 1e-9)) if mm.std() > 300 else 1.0
                dl = float((v.mean() - med.mean()) / rng_)
                if abs(g - 1) > 0.02 or abs(dl) > 0.01:
                    bad.add(k)
                    info["transition"][str(k)] = [round(g, 3), round(dl, 4)]
        for P in sorted(r["floor"]):
            rm = np.load(os.path.join(pdir, "ref", f"runmean_{r['r0']}_{P}.npy"))
            ph = r["phase"]
            # unscored pictures of this run with this phase: phase advances one class per frame
            k0 = next(iter(ph))
            p0 = int(ph[k0])
            seq = [1, 3, 5, 7]
            want = [k for k in range(r["r0"] + EDGE, r["r1"] - EDGE + 1) if k not in scored and k not in bad
                    and seq[(seq.index(p0) + (k - int(k0))) % 4] == int(P)]
            # nearest to the scored frames first
            mid = np.mean(r["stacked"])
            want = sorted(want, key=lambda k: abs(k - mid))
            got = []
            for k in want:
                fr = {n: [field(n, k, 0), field(n, k, 1)] for n in names}
                fr = {n: v for n, v in fr.items() if v[0] is not None and v[1] is not None}
                if len(fr) >= max(2, len(names) // 2):
                    got.append((k, fr))
                if len(got) >= a.frames // 4 + 1:
                    break
            if len(got) < 2:
                continue
            # 1. align every capture's fields of every frame to the median gains
            AL, GT = [], []
            for k, fr in got:
                alf = {}
                for fld in (0, 1):
                    ref = rm[fld][qm.RA, qm.CA].astype(np.float64)
                    prm = {n: fit(v[fld][0], ref) for n, v in fr.items()}
                    g_t = float(np.median([p[2] for p in prm.values()]))
                    o_t = float(np.median([p[3] for p in prm.values()]))
                    # chroma gain of a capture vs the mean is 1/|c|: the target is the captures' median
                    gc_t = float(np.median([1.0 / max(abs(p[1]), 1e-9) for p in prm.values()]))
                    al = {}
                    for n, v in fr.items():
                        dx, c, g, o = prm[n]
                        xs = qm.frac_shift(v[fld][0], -dx) if dx else v[fld][0]
                        z = qm.chroma_analytic(xs)
                        luma = (xs - z.real - o) / g * g_t + o_t
                        y = luma + (z * c * gc_t).real          # phase onto the mean, gain to the median
                        bad = np.array([yy in v[fld][1] for yy in range(qm.RA.start, qm.RA.stop)])
                        al[n] = (y, bad)
                    alf[fld] = al
                    GT.append(g_t)
                AL.append((k, alf))
            # 2. gross outliers: a capture's frame-to-frame random error (its residual against the
            #    median of the OTHER captures, differenced between consecutive frames) > OUTLIER x the
            #    median capture's
            ff = {}
            for t in range(len(AL) - 1):
                for fld in (0, 1):
                    a1, a2 = AL[t][1][fld], AL[t + 1][1][fld]
                    for n in a1:
                        if n not in a2 or len(a1) < 3:
                            continue
                        o1 = np.median([y for m, (y, _) in a1.items() if m != n], axis=0)
                        o2 = np.median([y for m, (y, _) in a2.items() if m != n], axis=0)
                        g = ~(a1[n][1] | a2[n][1])
                        if g.sum() < 20:
                            continue
                        d = (a1[n][0] - o1) - (a2[n][0] - o2)
                        # mean square: impulsive errors (unflagged speckle) count in full
                        ff.setdefault(n, []).append(float((d[g] ** 2).mean()))
            ffm = {n: float(np.median(v)) for n, v in ff.items()}
            med = float(np.median(list(ffm.values()))) if ffm else 0
            info.setdefault("ff_ratio", {})[P] = {n: round(v / med, 2) for n, v in ffm.items()} if med else {}
            drop = set(gross)

            # 3. mean of masters' means (captures equal within a master); a capture's dropout rows are
            #    left out of its master's mean there
            def combine(sel, fallback):
                ms = {}
                for n, (y, b) in sel.items():
                    ms.setdefault(master[n], []).append((y, b))
                mm = []
                for lst in ms.values():
                    num = sum(np.where(b[:, None], 0, y) for y, b in lst)
                    den = sum((~b).astype(float) for _, b in lst)[:, None]
                    mm.append(np.where(den > 0, num / np.maximum(den, 1), np.nan))
                with np.errstate(all="ignore"):
                    m = np.nanmean(mm, axis=0) if mm else fallback
                return np.where(np.isfinite(m), m, fallback)
            out, out0 = [], []
            for k, alf in AL:
                e, e0 = [], []
                for fld in (0, 1):
                    al = alf[fld]
                    keep = {n: v for n, v in al.items() if n not in drop}
                    fb = np.median([y for y, _ in keep.values()], axis=0)
                    e.append(combine(keep, fb))
                    e0.append(combine({n: v for n, v in keep.items() if master[n] == big}, fb))
                out.append(np.stack(e))
                out0.append(np.stack(e0))
            info["gain_median"][P] = float(np.median(GT))
            for lab, arr, fl in (("truth", out, "floor"), ("truth0", out0, "floor0")):
                arr = np.array(arr)
                mean = arr.mean(axis=0)
                full = np.array(rm, dtype=np.float32)
                full[:, qm.RA, qm.CA] = mean
                np.save(os.path.join(pdir, "ref", f"{lab}_{r['r0']}_{P}.npy"), full)
                ms = np.mean([[qm.band_ms(x[i] - mean[i]) for i in (0, 1)] for x in arr], axis=(0, 1))
                info[fl][P] = (ms / max(len(arr) - 1, 1)).tolist()
            info["frames"][P] = [k for k, _ in got]
        # consistency: each phase truth's LF luma gain against the median of the phase truths
        ph_lf = {}
        for P in info["floor"]:
            t = np.load(os.path.join(pdir, "ref", f"truth_{r['r0']}_{P}.npy"))[0][qm.RA, qm.CA].astype(np.float64)
            ph_lf[P] = qm.luma_lp(t[::4], 1.0)
        if len(ph_lf) >= 2:
            med = np.median(list(ph_lf.values()), axis=0)
            mm = med - med.mean()
            info["phase_gain"] = {P: round(float(((v - v.mean()) * mm).sum() / max((mm * mm).sum(), 1e-9)), 4)
                                  if mm.std() > 300 else 1.0 for P, v in ph_lf.items()}
        truth[str(r["r0"])] = info
        pg = info.get("phase_gain", {})
        worst = max((abs(g - 1) for g in pg.values()), default=0)
        line = (f"run {r['name']}@{r['r0']}: phases {sorted(info['frames'])}, frames "
                f"{sum(len(v) for v in info['frames'].values())}, excluded {info['excluded'] or 'none'}, "
                f"transition pictures dropped {sorted(int(k) for k in info['transition']) or 'none'}, "
                f"phase gains {pg}" + ("  !! phase gain off by > 1 %" if worst > 0.01 else ""))
        log.append(line)
        print(line, flush=True)
    for r0, info in truth.items():
        info["excluded"] = gross
    panel["truth"] = {"runs": truth, "masters": master, "excluded": gross, "largest_master": big, "log": log,
                      "method": "raw captures, median gain, masters equal, outliers > %gx dropped" % OUTLIER}
    out = a.out_json or os.path.join(pdir, "panel_v2.json")
    json.dump(panel, open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    print("wrote", out)


if __name__ == "__main__":
    main()
