#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast regression check of a stacker (and decoder) change on a fixed panel of pictures.

  quickcheck.py --stacker <stack.py | package dir> [--panel DIR] [--decoder t2d|v6|none]
                [--baseline RESULTS] [--out RESULTS] [--jobs N] [--sequential]

Stacks the panel's mini captures (see build_panel.sh) with the given stack.py and scores the
result in memory while the stacker is still writing it (frames are read back as they land, from
/dev/shm), then decodes the bars / test card / textured stills with ld-chroma-decoder transform2d
(several decoders at once, each frame scored as it arrives), optionally newdecode v6 on the key
frames, and writes <out>/results.json, crops.npz and contact.png (fixed crops, candidate | baseline |
difference).  With --baseline, prints the comparison with PASS/FAIL per check; exit 1 on any FAIL.
--sequential turns the overlaps off (stack, then score, then decode one range at a time) for timing.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qcmetrics as qm          # noqa: E402
import qcmap                    # noqa: E402
from qcpkg import make_pkg      # noqa: E402

PY = "/mnt/s/domsday/ldenv/bin/python"
MLPY = "/root/mlenv/bin/python"
ND = "/mnt/s/domsday/paldec_dev/v6/newdecode.py"
NDWRAP = "/mnt/e/domsday_scratch/fqv6/ndwrap_bench.py"
GPU_LOCK = "/run/lock/aiv-gpu.lock"
DEFAULT_PANEL = "/mnt/e/domsday_scratch/quickcheck/north2"
FBYTES = qm.H * qm.W * 2
OW, OH = 928, 576               # ld-chroma-decoder output (default active area, pad 8)

# thresholds: candidate vs baseline (ratio) and absolute weight sanity
TH_GRAIN = 1.10      # bars / flat grain and noise: not worse by > 10 %
TH_NOISE = 1.10
TH_DETAIL = 0.95     # detail: not lower by > 5 %
TH_FLICKER = 1.05    # flicker: not higher (5 % + floor for measurement jitter)
TH_W_EXCESS = 0.10   # blend error predicted from the measured master noise: <= 10 % above optimal
TH_W_DEV = 0.15      # no master's weight more than 0.15 above its noise-optimal weight
MIN_RUNS = 2         # weights are gated only when the master noise was measured on >= 2 held runs
COND_MIN = 0.05      # weights: a band is judged only where the master error covariance's smallest
                     # eigenvalue is >= 5 % of its largest variance
TH_MAP_RISE = 1.02   # map lines / text: 10-90 % rise width not wider by > 2 % (composite moves <= 1.3 % with
                     # noise, <= 3 % with +-0.7 px per-line jitter; secondary to the map error gate)
TH_MAP_TEXT = 0.98   # map text-area 2-5 MHz energy not lower by > 2 %
MIN_PIC_EDGES = 10       # the worst-picture rise gate uses pictures with >= 10 good edges (both runs)
TH_MAP_RISE_PIC = 1.03   # ... and no single map picture sharper-edged by less than 3 % / 5 %
# decoded pictures (t2d, newdecode): the decoders' own response to input noise moves the measured rise by
# 1.3-3.6 % at 1x a stack residual (test_rise_noise.py; the composite measure moves < 0.3 %), so wider
TH_MAP_RISE_DEC = 1.03
TH_MAP_RISE_PIC_DEC = 1.05
TH_MAP_TEXT_PIC = 0.95
TH_MAP_ERR = 1.02       # PRIMARY map gate: text / paper area error vs held-run truth
TH_GRAPHIC_RISE = 1.03   # bars / test card edges (not map lines): rise not wider by > 3 %
DARK_LEVEL = 17500   # mean active level below this: a black / near-black frame (black ~16000-16300)
APL_MIN = 22000      # flat-noise / flat-grain gates: frames with a lower mean level are left out (info)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- stack + log parse
def parse_log(text):
    out = {"captures": {}, "masters": [], "predicted": None}
    for m in re.finditer(r"^\s+(\S+)\s+noise=([\d.e+]+) master=(\S+) (\w+) \(([^)]*)\)", text, re.M):
        out["captures"][m.group(1)] = {"noise": float(m.group(2)), "master": m.group(3),
                                      "role": m.group(4), "reason": m.group(5)}
    for m in re.finditer(r"^\s+master (\d+) \((\d+) cap\)\s+([\d.\s]+?)\s+V ([\d\s]+?)\s+(?:picture/blank[\d.\s]+)?\[([^\]]*)\]",
                         text, re.M):
        out["masters"].append({"id": int(m.group(1)), "ncap": int(m.group(2)),
                               "w": [float(x) for x in m.group(3).split()],
                               "V": [float(x) for x in m.group(4).split()],
                               "names": [s.strip() for s in m.group(5).split(",")]})
    m = re.search(r"predicted blend error vs primary alone per band: (.*)", text)
    if m:
        out["predicted"] = m.group(1).strip()
    m = re.search(r"\((\d+) frames; (\d+) px cross-master filled; (\d+) px primary-master defects repaired", text)
    if m:
        out["frames"], out["filled_px"], out["repaired_px"] = map(int, m.groups())
    return out


def cav_picture(vbi):
    return qm.cav_picture(vbi)


# --------------------------------------------------------------------------- composite scoring
def truth_file(panel, pdir, r, P, which="truth"):
    """Held-run truth for run r, phase P: the raw-capture truth (build_truth.py) when the panel has it,
    else the reference stack's run mean."""
    t = (panel.get("truth") or {}).get("runs", {}).get(str(r["r0"]))
    if t and P in t.get("floor", {}):
        return os.path.join(pdir, "ref", f"{which}_{r['r0']}_{P}.npy")
    return os.path.join(pdir, "ref", f"runmean_{r['r0']}_{P}.npy")


def truth_floor(panel, r, key="floor"):
    t = (panel.get("truth") or {}).get("runs", {}).get(str(r["r0"]))
    if t and t.get(key):
        return t[key]
    return r["floor"]


class Scorer:
    def __init__(self, panel, pdir):
        self.p = panel
        self.pdir = pdir
        self.run_of = {}
        for r in panel.get("runs", []):
            for k in r["stacked"]:
                self.run_of[k] = r
        self.stills = {int(k): v for k, v in panel.get("stills", {}).items()}
        self.map = {int(k): v for k, v in panel.get("map", {}).items()}
        self.graphic = {int(k): v for k, v in panel.get("graphic", {}).items()}
        self._rm = {}
        self._lock = threading.Lock()
        rk = os.path.join(pdir, "ref", "ref_keys.npy")
        self.ref = None
        if os.path.exists(rk):
            self.ref_keys = {int(k): i for i, k in enumerate(np.load(rk))}
            self.ref = np.load(os.path.join(pdir, "ref", "ref.npy"), mmap_mode="r")

    def runmean(self, r, P, which="truth"):
        """The run's truth for PAL phase class P (fieldPhaseID of the frame's first field)."""
        with self._lock:
            if (r["r0"], P, which) not in self._rm:
                self._rm[r["r0"], P, which] = np.load(truth_file(self.p, self.pdir, r, P, which))
            return self._rm[r["r0"], P, which]

    def _caps(self):
        with self._lock:
            if not hasattr(self, "_capsrc"):
                self._capsrc = []
                for n in self.p["caps"]:
                    b = os.path.join(self.pdir, "caps", n)
                    try:
                        f = json.load(open(b + ".tbc.json"))["fields"]
                    except OSError:
                        continue
                    idx = {k: j for j, k in enumerate(qm.frame_keys(f, self.p.get("clv", False))) if k is not None}
                    self._capsrc.append((idx, np.memmap(b + ".tbc", "<u2", "r").reshape(-1, 2, qm.H, qm.W)))
            return self._capsrc

    def cap_gain(self, key, a):
        """LF (< 1 MHz) luma gain of an output frame against the median of the raw captures' frames of
        the same picture: < 1 means the output lost contrast (and so looks cleaner than it is)."""
        cs = [tb[idx[key]] for idx, tb in self._caps() if key in idx]
        if len(cs) < 1:
            return None
        m = np.median([c[:, qm.RA, qm.CA].astype(np.float64) for c in cs], axis=0)
        g = []
        for i in (0, 1):
            x = qm.luma_lp(a[i] - a[i].mean(), 1.0)
            y = qm.luma_lp(m[i] - m[i].mean(), 1.0)
            if (y * y).sum() > 0 and y.std() > 300:
                g.append(float((x * y).sum() / (y * y).sum()))
        return float(np.mean(g)) if g else None

    def frame(self, key, fr):
        f = fr.astype(np.float64)
        out = {"key": key}
        a = f[:, qm.RA, qm.CA]
        out["lvl"] = [float(a[0].mean()), float(a[1].mean())]
        if key in self.stills:
            out["cap_gain"] = self.cap_gain(key, a)
        if self.ref is not None and key in self.ref_keys:
            d = fr.astype(np.int32) - self.ref[self.ref_keys[key]].astype(np.int32)
            da = d[:, qm.RA, qm.CA]
            out["vs_prod"] = {"rms": float(np.sqrt((da.astype(np.float64) ** 2).mean())),
                              "exact": float((d == 0).mean()), "maxabs": int(np.abs(d).max())}
        r = self.run_of.get(key)
        P = str(r["phase"][str(key)]) if r is not None else None
        if r is not None and P in truth_floor(self.p, r):
            rm = self.runmean(r, P)[:, qm.RA, qm.CA]
            if os.path.exists(truth_file(self.p, self.pdir, r, P, "truth0")) and "truth" in self.p:
                e0 = a - self.runmean(r, P, "truth0")[:, qm.RA, qm.CA]
                out["err0_ms"] = [qm.band_ms(e0[i]).tolist() for i in (0, 1)]
            out["phase"] = P
            e = a - rm
            out["run"] = r["r0"]
            out["err_ms"] = [qm.band_ms(e[i]).tolist() for i in (0, 1)]
            out["lvl_off"] = float(e.mean())
            z = qm.chroma_analytic(a)
            zr = qm.chroma_analytic(rm)
            c = (z * np.conj(zr)).sum()
            out["hue"] = float(np.degrees(np.angle(c)))
            out["sat"] = float(np.abs(c) / (np.abs(zr) ** 2).sum())
            out["chroma_amp"] = float(np.sqrt((np.abs(zr) ** 2).mean()))
            ab, bb = qm.band_cross(a.reshape(-1, a.shape[-1]), rm.reshape(-1, rm.shape[-1]))
            out["gain_ab"], out["gain_bb"] = ab.tolist(), bb.tolist()
        if key in self.map:
            out["map_comp"] = qcmap.measure_comp(fr, self.map[key])
            rr = self.run_of.get(key)
            Pm = str(rr["phase"][str(key)]) if rr is not None else None
            if rr is not None and Pm in truth_floor(self.p, rr):
                tr = self.runmean(rr, Pm)
                sp = self.map[key]["comp"]
                fl = np.array(truth_floor(self.p, rr)[Pm])
                out["map_err"] = {lab: (np.mean([qm.band_ms((f[ff] - tr[ff])[y:y + 16, x:x + 64]) for ff, y, x in sp[lab]],
                                                axis=0) - fl).tolist() for lab in ("text", "paper") if sp[lab]}
        if key in self.graphic:
            out["graphic_comp"] = qcmap.measure_comp(fr, self.graphic[key])
        s = self.stills.get(key)
        if s is not None:
            fl = [qm.blocks_ms(f[i], [(y, x) for ff, y, x in s["flat"] if ff == i]) for i in (0, 1)]
            tx = [qm.blocks_ms(f[i], [(y, x) for ff, y, x in s["tex"] if ff == i]) for i in (0, 1)]
            out["flat_ms"] = np.nanmean(fl, axis=0).tolist()
            out["tex_ms"] = np.nanmean(tx, axis=0).tolist()
        return out


def summarise(panel, frames, meta):
    """Aggregate per-frame measurements into the scored metrics."""
    S = {}
    runs = {r["r0"]: r for r in panel.get("runs", [])}
    fr = [f for f in frames.values() if "err_ms" in f]
    if fr:
        ms = np.array([np.mean(f["err_ms"], axis=0) - np.array(truth_floor(panel, runs[f["run"]])[f["phase"]]) for f in fr])
        S["err_run"] = np.sqrt(np.maximum(ms.mean(axis=0), 0)).tolist()
        f0 = [f for f in fr if "err0_ms" in f]
        if f0:
            ms0 = np.array([np.mean(f["err0_ms"], axis=0) - np.array(truth_floor(panel, runs[f["run"]], "floor0")[f["phase"]])
                            for f in f0])
            S["err_run_truth0"] = np.sqrt(np.maximum(ms0.mean(axis=0), 0)).tolist()
        lv, hu, sa, ga = [], [], [], []
        for r0, r in runs.items():
            g = [frames[k] for k in r["stacked"] if k in frames and "err_ms" in frames[k]]
            if len(g) < 2:
                continue
            lv.append(np.std([x["lvl_off"] for x in g], ddof=1))
            if np.mean([x["chroma_amp"] for x in g]) > 1000:      # hue / saturation only where there is colour
                hu.append(np.std([x["hue"] for x in g], ddof=1))
                sa.append(100 * np.std([x["sat"] for x in g], ddof=1))
            if r.get("textured"):
                ga.append(np.sum([x["gain_ab"] for x in g], axis=0) / np.sum([x["gain_bb"] for x in g], axis=0))
        S["flicker_level"] = float(np.mean(lv)) if lv else None
        S["flicker_hue_deg"] = float(np.mean(hu)) if hu else None
        S["flicker_sat_pct"] = float(np.mean(sa)) if sa else None
        S["bias_level"] = float(np.mean([abs(np.mean([frames[k]["lvl_off"] for k in r["stacked"] if k in frames]))
                                         for r in runs.values() if any(k in frames for k in r["stacked"])]))
        if ga:
            S["gain_run"] = np.mean(ga, axis=0).tolist()
    st = [f for f in frames.values() if "flat_ms" in f]
    if st:
        # black / near-black frames apart from pictures: a stacker can treat dark frames differently
        # near-black / low-APL frames apart: there a gain-squashing fit looks clean (EcoDisc 53924/7)
        dark = [f for f in st if np.mean(f["lvl"]) < APL_MIN]
        pic = [f for f in st if np.mean(f["lvl"]) >= APL_MIN]
        # noise relative to the output's own contrast against the raw captures
        gn = lambda f: (f.get("cap_gain") or 1.0) if 0.5 < (f.get("cap_gain") or 1.0) < 2 else 1.0
        pic = [dict(f, flat_ms=(np.array(f["flat_ms"]) / gn(f) ** 2).tolist()) for f in pic]
        if pic:
            S["noise_flat"] = np.sqrt(np.nanmean([f["flat_ms"] for f in pic], axis=0)).tolist()
            fl = np.array([f["flat_ms"] for f in pic])
            tx = np.array([f["tex_ms"] for f in pic])
            S["detail_tex"] = np.sqrt(np.maximum(np.nanmean(tx - fl, axis=0), 0)).tolist()
        if dark:
            S["noise_black"] = np.sqrt(np.nanmean([f["flat_ms"] for f in dark], axis=0)).tolist()
    vp = [f["vs_prod"] for f in frames.values() if "vs_prod" in f]
    if vp:
        S["vs_prod_rms"] = float(np.mean([v["rms"] for v in vp]))
        S["vs_prod_exact_frames"] = int(sum(v["maxabs"] == 0 for v in vp))
        S["vs_prod_frames"] = len(vp)
        S["vs_prod_max"] = int(max(v["maxabs"] for v in vp))
    mc = [f["map_comp"] for f in frames.values() if "map_comp" in f]
    if mc:
        S["map_comp"] = qcmap.summarise(mc)
    S.update(map_err_summary(frames.values(), "map_err_comp"))
    gc = [f["graphic_comp"] for f in frames.values() if "graphic_comp" in f]
    if gc:
        S["graphic_comp"] = qcmap.summarise(gc)
    if meta is not None:
        S.update(meta)
    return S


def _gate_frame(frames, k):
    """(use for flat grain?, gain) of picture k from the composite measurements."""
    f = (frames or {}).get(int(k)) or (frames or {}).get(str(k))
    if not f:
        return True, 1.0
    g = f.get("cap_gain") or 1.0
    return np.mean(f["lvl"]) >= APL_MIN, (g if 0.5 < g < 2 else 1.0)


def _flat_grain(dd, frames):
    v = []
    for k, x in dd.items():
        if not x.get("y_flat_grain"):
            continue
        use, g = _gate_frame(frames, x.get("key", k))
        if use:
            v.append((x["y_flat_grain"] / g) ** 2)
    return float(np.sqrt(np.mean(v))) if v else None


def map_err_summary(vals, key):
    """rms error vs the held-run truth in map text / line blocks and paper blocks, over the map pictures
    that sit in held runs: {key: {"text": rms, "paper": rms, "pictures": n, "bands_text": [...]}}."""
    me = [v["map_err"] if "map_err" in v else v.get("map_err_dec") for v in vals]
    me = [m for m in me if m]
    if not me:
        return {}
    out = {"pictures": len(me)}
    for lab in ("text", "paper"):
        b = [m[lab] for m in me if lab in m]
        if b:
            ms = np.maximum(np.mean(b, axis=0), 0)
            out[lab] = float(np.sqrt(ms.sum()))
            out["bands_" + lab] = np.sqrt(ms).tolist()
    return {key: out}


def decoded_summary(dres, frames=None):
    """Aggregate the transform2d measurements (per frame, from score_decoded)."""
    S = {}
    bg = [x for v in dres.values() for x in v.get("bar_grain", [])]
    if bg:
        S["bar_grain"] = {str(v["key"]): v["bar_grain"] for v in dres.values() if "bar_grain" in v}
        S["bar_grain_mean"] = float(np.mean(bg))
        S["bar_grain_max"] = float(np.max(bg))
    yf = [v for v in dres.values() if "y_flat_ms" in v]
    if yf:
        fl = np.nanmean([v["y_flat_ms"] for v in yf], axis=0)
        tx = np.nanmean([v["y_tex_ms"] for v in yf], axis=0)
        S["y_detail"] = np.sqrt(np.maximum(tx - fl, 0)).tolist()
        S["y_flat_grain"] = _flat_grain(dres, frames)
    md = [v["map_dec"] for v in dres.values() if "map_dec" in v]
    if md:
        S["map_dec"] = qcmap.summarise(md)
    S.update(map_err_summary(dres.values(), "map_err_dec"))
    gd = [v["graphic_dec"] for v in dres.values() if "graphic_dec" in v]
    if gd:
        S["graphic_dec"] = qcmap.summarise(gd)
    cb = [v["comb"] for v in dres.values() if "comb" in v]
    if cb:
        S["comb_t2d"] = float(np.mean(cb))
    return S


def resummarise(panel, res):
    """A results.json's summary recomputed from its per-frame data with this version's metrics, so
    a baseline scored by an older quickcheck still compares like with like."""
    if not res.get("frames"):
        return res["summary"]
    old = res["summary"]
    frames = {int(k): v for k, v in res["frames"].items()}
    S = summarise(panel, frames, {k: old[k] for k in ("resid_do", "resid_do_frames", "out_frames") if k in old})
    if isinstance(res.get("decoded"), dict):
        S.update(decoded_summary(res["decoded"], frames))
    if isinstance(res.get("newdecode"), dict):
        nd = decoded_summary(res["newdecode"], frames)
        if "map_dec" in nd:
            S["map_v6"] = nd["map_dec"]
    for k in ("nd_bar_grain_mean", "nd_flat_grain"):
        if k in old:
            S[k] = old[k]
    return S


# --------------------------------------------------------------------------- master weights check
def _base_name(n):
    m = re.match(r"^(.*ds\d+)[a-z]$", n)
    return m.group(1) if m else n


def _psd(M):
    """Nearest positive semi-definite matrix (eigenvalues clipped at 0)."""
    M = 0.5 * (M + M.T)
    w, v = np.linalg.eigh(M)
    return (v * np.maximum(w, 0)) @ v.T


def weights_check(panel, lg):
    """Blend weights from the log against the per-master noise measured on the panel's held runs
    (capture-vs-run-mean residual covariance, independent of any stacker).  Split captures (ds1a /
    ds1b) count as one; captures without enough measured overlap are left out; the master error
    covariance is projected onto the PSD cone.  {"status": "insufficient data", "why": ...} when the
    panel cannot support the estimate (a master with no measured capture, a missing pair, or a
    master whose estimated error is not positive)."""
    mc = panel.get("master_cov")
    if not lg["masters"]:
        return None
    if not mc:
        return {"status": "insufficient data", "why": "no master_cov"}
    names = mc["names"]
    valid = set(mc.get("valid", names))
    C = np.array([[[np.nan if x is None else x for x in row] for row in band] for band in mc["C"]], float)
    runs_note = None
    if mc.get("per_run"):
        # a capture missing a held run would be measured on other content than the rest: use the set of
        # runs that every capture kept has, choosing the set that keeps most masters, then reaches
        # MIN_RUNS runs, then keeps most captures
        import itertools
        pr = {k: {f: np.array(v[f], float) for f in ("S", "F", "N")} for k, v in mc["per_run"].items()}
        lg_groups = [list(dict.fromkeys(_base_name(n) for n in m["names"])) for m in lg["masters"]]
        best = None
        for L in range(len(pr), 0, -1):
            for sub in itertools.combinations(sorted(pr), L):
                Nn = sum(pr[k]["N"] for k in sub)
                ok = [n for n in names if all(pr[k]["N"][names.index(n), names.index(n)] >= 2 for k in sub)
                      and Nn[names.index(n), names.index(n)] >= 4]
                cover = sum(any(n in ok for n in g) for g in lg_groups)
                # all masters first, then two runs (needed to gate), then captures, then runs
                score = (cover, min(L, MIN_RUNS), len(ok), L)
                if best is None or score > best[0]:
                    best = (score, sub, ok)
        _, sub, ok = best
        Ss = sum(pr[k]["S"] for k in sub)
        Fs = sum(pr[k]["F"] for k in sub)
        Nn = sum(pr[k]["N"] for k in sub)
        Nn = Nn + Nn.T - np.diag(np.diag(Nn))
        Ss = Ss + np.transpose(Ss, (0, 2, 1)) * (1 - np.eye(len(names)))
        Fs = Fs + np.transpose(Fs, (0, 2, 1)) * (1 - np.eye(len(names)))
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.where(Nn >= 4, (Ss - Fs) / np.maximum(Nn, 1), np.nan)
        valid = set(ok) & valid if valid else set(ok)
        valid = set(ok)
        runs_note = list(sub)
    C = C - np.array(mc.get("common", [0.0] * len(qm.BANDS)))[:, None, None]
    groups, capw = [], {}
    for m in lg["masters"]:
        g = list(dict.fromkeys(_base_name(n) for n in m["names"]))
        g = [n for n in g if n in names and n in valid]
        if not g:
            return {"status": "insufficient data", "why": f"master {m['id']} ({', '.join(m['names'])}): no capture "
                    "with enough clean overlap on the held runs", "groups": [m["names"] for m in lg["masters"]]}
        groups.append(g)
    for n, c in lg["captures"].items():
        b = _base_name(n)
        capw[b] = max(capw.get(b, 0), 1.0 / max(c.get("noise", 1.0), 1e-6) ** 2)
    # pairs never measured together (a capture missing a held run): leave out the capture with the
    # most missing pairs until the rest is complete, keeping at least one capture per master
    dropped = []
    while True:
        used = [n for g in groups for n in g]
        ix = [names.index(n) for n in used]
        bad = ~np.isfinite(C[:, ix][:, :, ix]).all(axis=0)
        if not bad.any():
            break
        cnt = bad.sum(axis=1)
        order = [used[i] for i in np.argsort(-cnt) if cnt[i] > 0]
        n = next((n for n in order if len(next(g for g in groups if n in g)) > 1), None)
        if n is None:
            return {"status": "insufficient data", "why": "captures of different masters never overlap on the held runs",
                    "groups": groups}
        for g in groups:
            if n in g:
                g.remove(n)
        dropped.append(n)
    nb = C.shape[0]
    res = {"status": "ok", "V": [], "w_log": [], "w_opt": [], "excess": [], "dev": [], "groups": groups,
           "left_out": dropped, "bands_na": [], "runs": runs_note}
    for b in range(nb):
        A = np.zeros((len(groups), len(names)))
        for m, g in enumerate(groups):
            for n in g:
                A[m, names.index(n)] = capw.get(n, 1.0)
            A[m] /= max(A[m].sum(), 1e-30)
        M = _psd(A @ np.nan_to_num(C[b]) @ A.T)          # master estimate error covariance
        ev = np.linalg.eigvalsh(M)
        if ev.min() < COND_MIN * np.diag(M).max():
            # a master's error not positive, or masters' errors (nearly) identical once the shared part
            # is removed: the optimum is not determined by this measurement
            res["bands_na"].append(qm.BAND_NAMES[b])
            for k in ("V", "w_log", "w_opt"):
                res[k].append(None)
            res["excess"].append(None)
            res["dev"].append(None)
            continue
        wl = np.array([lg["masters"][m]["w"][b] for m in range(len(groups))])
        wl = wl / wl.sum()
        try:
            wo = np.linalg.solve(M + 1e-9 * np.trace(M) * np.eye(len(M)), np.ones(len(groups)))
        except np.linalg.LinAlgError:
            wo = 1.0 / np.diag(M)
        wo = np.clip(wo, 0, None)
        wo = wo / wo.sum()
        vl, vo = wl @ M @ wl, wo @ M @ wo
        res["V"].append(np.diag(M).tolist())
        res["w_log"].append(wl.tolist())
        res["w_opt"].append(wo.tolist())
        res["excess"].append(float(vl / vo - 1))
        res["dev"].append(float(np.max(wl - wo)))
    ok = [b for b in range(nb) if res["excess"][b] is not None]
    if not ok:
        return {"status": "insufficient data", "why": "no band where the masters' errors are separable: "
                + ", ".join(res["bands_na"]), "groups": groups}
    res["max_excess"] = max(res["excess"][b] for b in ok)
    res["max_dev"] = max(res["dev"][b] for b in ok)
    res["pass"] = bool(res["max_excess"] <= TH_W_EXCESS and res["max_dev"] <= TH_W_DEV)
    nr = len(runs_note) if runs_note is not None else len(mc.get("runs_used") or [0, 0])
    res["runs_used"] = runs_note or mc.get("runs_used")
    if nr < MIN_RUNS:
        res["status_note"] = f"measured on {nr} held run(s) only: not gated"
    return res


# --------------------------------------------------------------------------- decode stage
def yuv_planes(buf):
    a = np.frombuffer(buf, "<u2").reshape(3, OH, OW)
    return a


def to_rgb8(yuv):
    y = (yuv[0].astype(np.float32) - 4096) / (60160 - 4096)
    u = (yuv[1].astype(np.float32) - 32768) / (57344 - 8192)
    v = (yuv[2].astype(np.float32) - 32768) / (57344 - 8192)
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255 + 0.5).astype(np.uint8)


def _do_mask(fields, j, mask, grow=2):
    """OR frame j's flagged dropouts (fields 2j, 2j+1 of a .tbc.json list) into mask [2,H,W]."""
    for f in (0, 1):
        do = fields[2 * j + f].get("dropOuts") or {}
        for ln, sx, ex in zip(do.get("fieldLine", []), do.get("startx", []), do.get("endx", [])):
            y = int(ln) - 1                     # fieldLine is 1-based
            if 0 <= y < qm.H:
                mask[f, y, max(int(sx) - grow, 0):min(int(ex) + grow, qm.W)] = True


def dropout_error(panel, pdir, stacks):
    """rms error vs the held-run truth at the dropout pixels of the scored held-run frames, for each
    stack in `stacks` ({label: stack base}), all on the SAME pixels: those flagged on any capture or
    by any of the stacks (old and new flagging conventions differ, so the residual count does not
    compare; this does)."""
    clv = panel.get("clv", False)
    capm = []
    for n in panel["caps"]:
        f = json.load(open(os.path.join(pdir, "caps", n + ".tbc.json")))["fields"]
        capm.append((f, {k: j for j, k in enumerate(qm.frame_keys(f, clv)) if k is not None}))
    stm = {}
    for lab, b in stacks.items():
        f = json.load(open(b + ".tbc.json"))["fields"]
        stm[lab] = (f, {k: j for j, k in enumerate(qm.frame_keys(f, clv)) if k is not None},
                    np.memmap(b + ".tbc", "<u2", "r").reshape(-1, 2, qm.H, qm.W))
    acc = {lab: [0.0, 0] for lab in stacks}
    for r in panel.get("runs", []):
        for k in r["stacked"]:
            P = str(r["phase"].get(str(k)))
            if P not in truth_floor(panel, r) or not all(k in m[1] for m in stm.values()):
                continue
            mask = np.zeros((2, qm.H, qm.W), bool)
            for f, idx in capm:
                if k in idx:
                    _do_mask(f, idx[k], mask)
            for f, idx, _ in stm.values():
                _do_mask(f, idx[k], mask)
            act = np.zeros_like(mask)
            act[:, qm.RA, qm.CA] = True
            mask &= act
            if not mask.any():
                continue
            rm = np.load(truth_file(panel, pdir, r, P))
            for lab, (f, idx, tb) in stm.items():
                e = tb[idx[k]].astype(np.float64)[mask] - rm[mask]
                acc[lab][0] += float((e * e).sum())
                acc[lab][1] += int(mask.sum())
    return {lab: ({"rms": float(np.sqrt(v[0] / v[1])), "pixels": v[1]} if v[1] else None) for lab, v in acc.items()}


def comb_index(Y):
    """Interlace combing: line-to-line (other field) roughness over same-field roughness of the decoded
    luma; ~1-2 on a clean frame, much higher where the two fields of a moving picture disagree."""
    y = Y[20:-20].astype(np.float64)
    e1 = ((y[1:-1] - 0.5 * (y[:-2] + y[2:])) ** 2).mean()
    e2 = ((y[2:-2] - 0.5 * (y[:-4] + y[4:])) ** 2).mean()
    return float(e1 / max(e2, 1e-9))


def ghost_coef(Yd, Yk, Yp, Yn):
    """How far a decoded motion frame Yd leans toward its neighbours, measured against the
    transform2d frames of the same stack (Yk itself, Yp / Yn before / after): 0 = none, 1 = the mean
    of the neighbours (a held-still average applied to motion)."""
    N = 0.5 * (Yp.astype(np.float64) + Yn) - Yk
    return float(((Yd.astype(np.float64) - Yk) * N).sum() / max((N * N).sum(), 1e-9))


AUDIO_WIN = 44100


def audio_metrics(stkbase, work, ld="/mnt/s/domsday/ld-decode"):
    """Production audio chain on the stack's .pcm (squelch_pcm.py, then mono_pcm.py with the stack's
    VBI): per 1 s window, programme = correlated L/R (|r| > 0.9) at rms > 200 in the stack's audio;
    muted = squelched rms < 10 % of it; mono = output L == R."""
    pcm = stkbase + ".pcm"
    if not os.path.exists(pcm) or os.path.getsize(pcm) < 4 * AUDIO_WIN:
        return None
    sq, mo = os.path.join(work, "sq.pcm"), os.path.join(work, "mono.pcm")
    subprocess.run([PY, os.path.join(ld, "scripts", "squelch_pcm.py"), pcm, sq, "--json", stkbase + ".tbc.json"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(sq):
        return None
    subprocess.run([PY, os.path.join(ld, "scripts", "mono_pcm.py"), sq, mo, "--json=" + stkbase + ".tbc.json"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rd = lambda f: np.fromfile(f, "<i2").reshape(-1, 2).astype(np.float64)
    a, b = rd(pcm), rd(sq)
    m = rd(mo) if os.path.exists(mo) else b
    n = min(len(a), len(b), len(m)) // AUDIO_WIN
    prog = muted = mono = wrong = 0
    for w in range(n):
        x = a[w * AUDIO_WIN:(w + 1) * AUDIO_WIN]
        rms = np.sqrt((x ** 2).mean())
        r = np.corrcoef(x[:, 0], x[:, 1])[0, 1] if x.std(axis=0).min() > 0 else 0
        p = rms > 200 and abs(r) > 0.9
        mu = np.sqrt((b[w * AUDIO_WIN:(w + 1) * AUDIO_WIN] ** 2).mean()) < 0.1 * rms
        y = m[w * AUDIO_WIN:(w + 1) * AUDIO_WIN]
        mn = np.abs(y[:, 0] - y[:, 1]).max() < 2 and np.abs(y).max() > 0
        prog += p
        muted += p and mu
        mono += mn
        wrong += mn and abs(r) < 0.8 and rms > 200
    return {"seconds": n, "programme_s": int(prog), "programme_muted_s": int(muted), "mono_s": int(mono),
            "mono_uncorrelated_s": int(wrong)}


def crop_rgb(yuv, c):
    r = to_rgb8(yuv)[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
    k = int(c.get("scale", 1))
    return np.repeat(np.repeat(r, k, axis=0), k, axis=1) if k > 1 else r


_TRUTH_DEC = {}


def truth_decoded(panel, key):
    """transform2d luma of the held-run truth for picture key (cached in ref/truthdec_*.npy), or None
    when key is not in a held run.  Decoded outputs (t2d and newdecode) are compared with it."""
    pdir = panel.get("_dir")
    for r in panel.get("runs", []):
        if key in r["stacked"]:
            P = str(r["phase"][str(key)])
            if P not in truth_floor(panel, r) or not pdir:
                return None
            f = os.path.join(pdir, "ref", f"truthdec_{r['r0']}_{P}_{key}.npy")
            if f in _TRUTH_DEC:
                return _TRUTH_DEC[f]
            if not os.path.exists(f):
                import qcrender
                from extract_panel import ref_pic2frame
                tr = np.clip(np.load(truth_file(panel, pdir, r, P)), 0, 65535).astype(np.uint16)
                meta = qcrender.load_meta(panel["reference"]["tbc"])
                p2f = ref_pic2frame(panel["reference"])
                with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
                    qcrender.frames_tbc(None, meta, [p2f[key]], os.path.join(td, "t"), frames=tr[None])
                    y = qcrender.decode(os.path.join(td, "t"))[0][0].astype(np.float32)
                np.save(f + ".tmp.npy", y)
                os.replace(f + ".tmp.npy", f)
            _TRUTH_DEC[f] = np.load(f).astype(np.float64)
            return _TRUTH_DEC[f]
    return None


def score_decoded(panel, key, yuv):
    d = panel["decode"]
    out = {"key": key}
    k = str(key)
    Y = yuv[0].astype(np.float64)
    if k in d.get("bar_regions", {}):
        br = d["bar_regions"][k]
        y0, y1 = br["rows"]
        g = []
        for x0, x1 in br["bars"]:
            blk = Y[y0:y1, x0:x1]
            blk = blk - blk.mean(axis=1, keepdims=True)
            blk = blk - blk.mean(axis=0, keepdims=True)
            g.append(float(np.sqrt((blk ** 2).mean())))
        out["bar_grain"] = g
    if k in panel.get("map", {}):
        out["map_dec"] = qcmap.measure_dec(yuv[0], panel["map"][k])
        td = truth_decoded(panel, key)
        if td is not None:
            Y = yuv[0].astype(np.float64)
            sp = panel["map"][k]["dec"]
            out["map_err_dec"] = {lab: np.mean([qm.band_ms((Y - td)[y:y + 32, x:x + 64]) for par, y, x in sp[lab]],
                                               axis=0).tolist() for lab in ("text", "paper") if sp[lab]}
    if k in panel.get("graphic", {}):
        out["graphic_dec"] = qcmap.measure_dec(yuv[0], panel["graphic"][k])
    if key in set(panel.get("motion", [])):
        out["comb"] = comb_index(yuv[0])
    if k in d.get("tex_blocks", {}):
        tb = d["tex_blocks"][k]

        def bms(blocks):
            v = []
            for par, y, x in blocks:
                blk = Y[y + par:y + 32:2, x:x + 64]
                blk = blk - blk.mean(axis=1, keepdims=True)
                v.append(qm.band_ms(blk))
            return np.mean(v, axis=0) if v else np.full(4, np.nan)
        fl, tx = bms(tb["flat"]), bms(tb["tex"])
        out["y_flat_ms"] = fl.tolist()
        out["y_tex_ms"] = tx.tolist()
        # grain: luma rms after row/col mean removal in the flat blocks
        gr = []
        for par, y, x in tb["flat"]:
            blk = Y[y + par:y + 32:2, x:x + 64]
            blk = blk - blk.mean(axis=1, keepdims=True)
            blk = blk - blk.mean(axis=0, keepdims=True)
            gr.append((blk ** 2).mean())
        out["y_flat_grain"] = float(np.sqrt(np.mean(gr))) if gr else None
    return out


def decode_ranges(tbc, frames_needed, sequential, threads=4, par=4):
    """{output frame index (0-based): yuv uint16[3,OH,OW]} via ld-chroma-decoder transform2d; one
    process per contiguous range, `par` at once, each read as it streams."""
    idx = sorted(frames_needed)
    ranges = []
    for i in idx:
        if ranges and i == ranges[-1][1] + 1:
            ranges[-1][1] = i
        else:
            ranges.append([i, i])
    fb = OW * OH * 3 * 2
    res = {}

    def one(r):
        a, b = r
        p = subprocess.Popen(["ld-chroma-decoder", "-q", "-f", "transform2d", "-t", str(threads), "-p", "yuv",
                              "-s", str(a + 1), "-l", str(b - a + 1), "--input-json", tbc + ".json", tbc, "-"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=fb)
        got = {}
        for i in range(a, b + 1):
            buf = p.stdout.read(fb)
            if len(buf) < fb:
                break
            got[i] = yuv_planes(buf)
        p.wait()
        return got

    if sequential:
        for r in ranges:
            res.update(one(r))
    else:
        with cf.ThreadPoolExecutor(par) as ex:
            for g in ex.map(one, ranges):
                res.update(g)
    return res


class _Done:
    """Stands in for the stacker process with --from-stack."""
    pid = -1

    def poll(self):
        return 0

    def wait(self):
        return 0


def nd_running():
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any("nd_gpuwrap" in ln for ln in ps.splitlines())


def watch_rss(proc, peak):
    """Peak summed PSS of the stacker and its workers."""
    while proc.poll() is None:
        tot = 0
        pids = [proc.pid]
        try:
            for d in os.listdir("/proc"):
                if d.isdigit():
                    try:
                        st = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
                        if int(st[1]) == proc.pid:
                            pids.append(int(d))
                    except (OSError, IndexError, ValueError):
                        pass
            for pid in pids:
                try:
                    # PSS: forked workers share the parent's pages; RSS would count them per worker
                    for ln in open(f"/proc/{pid}/smaps_rollup"):
                        if ln.startswith("Pss:"):
                            tot += int(ln.split()[1]) * 1024
                except OSError:
                    pass
        except OSError:
            pass
        peak[0] = max(peak[0], tot)
        time.sleep(2)


# --------------------------------------------------------------------------- newdecode (optional)
def gpu_free():
    """No production newdecode (nd_gpuwrap) running.  Other GPU users are serialised by running
    newdecode under flock on aiv-gpu.lock (v6_decode)."""
    return not nd_running()


ND_SNAP = "/mnt/e/domsday_scratch/quickcheck/ndpkg"


def nd_hash(path):
    """Hash of a newdecode: every .py / .json beside newdecode.py (optcore, model_params, ...)."""
    import hashlib
    d = os.path.dirname(os.path.abspath(path))
    h = hashlib.md5()
    for n in sorted(os.listdir(d)):
        if n.endswith((".py", ".json")):
            h.update(n.encode())
            h.update(open(os.path.join(d, n), "rb").read())
    return h.hexdigest()[:12]


def nd_script(d):
    """--decoder value -> newdecode.py of a frozen copy (ndpkg/<hash>/: the directory under test may be
    edited while a check runs), or None for t2d / none."""
    if d in (None, "t2d", "none"):
        return None
    p = ND if d == "v6" else os.path.abspath(d)
    if os.path.isdir(p):
        p = os.path.join(p, "newdecode.py")
    h = nd_hash(p)
    dst = os.path.join(ND_SNAP, h)
    if not os.path.exists(os.path.join(dst, "newdecode.py")):
        tmp = dst + ".tmp%d" % os.getpid()
        os.makedirs(tmp, exist_ok=True)
        src = os.path.dirname(p)
        for n in os.listdir(src):
            if n.endswith((".py", ".json")):
                shutil.copy2(os.path.join(src, n), os.path.join(tmp, n))
        try:
            os.rename(tmp, dst)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    return os.path.join(dst, "newdecode.py")


def nd_pictures(k2out, keys, ctx):
    """Pictures to decode: each key picture with its consecutive neighbours (consecutive both in
    picture number and in the stack), up to ctx either side."""
    pics = set()
    for k in keys:
        if k not in k2out:
            continue
        lo = hi = k
        while k - lo < ctx and lo - 1 in k2out and k2out[lo - 1] == k2out[lo] - 1:
            lo -= 1
        while hi - k < ctx and hi + 1 in k2out and k2out[hi + 1] == k2out[hi] + 1:
            hi += 1
        pics.update(range(lo, hi + 1))
    return sorted(pics)


def _md5(path):
    import hashlib
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def v6_decode(panel, stkbase, meta, work, log_to, ctx, cache_dir=None, lock_wait=1200, nd=ND):
    """{key picture: yuv} of newdecode v6 over a small TBC of the key pictures and their runs
    (ld-dropout-correct first, as production).  cache_dir: reuse / store the result there, keyed by
    newdecode + wrapper md5 and the picture list."""
    import hashlib
    flds = meta["fields"]
    k2out = {}
    for j, k in enumerate(qm.frame_keys(flds, panel.get("clv", False))):
        if k is not None and k not in k2out:
            k2out[k] = j
    keys = [k for k in panel["decode"]["nd_frames"] if k in k2out]
    pics = nd_pictures(k2out, keys, ctx)
    cache = None
    if cache_dir:
        tag = hashlib.md5(json.dumps([nd_hash(nd), _md5(NDWRAP), pics, keys]).encode()).hexdigest()[:12]
        cache = os.path.join(cache_dir, f"v6cache_{tag}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            log(f"[qc] v6 baseline from cache {cache}")
            return {int(k): z[k] for k in z.files}
    os.makedirs(work, exist_ok=True)
    out_idx = [k2out[p] for p in pics]
    sub = os.path.join(work, "nd_in")
    with open(stkbase + ".tbc", "rb") as f, open(sub + ".tbc", "wb") as o:
        for i in out_idx:
            f.seek(i * 2 * FBYTES)
            o.write(f.read(2 * FBYTES))
    m = dict(meta)
    m["fields"] = [dict(flds[2 * i + k], seqNo=2 * j + k + 1) for j, i in enumerate(out_idx) for k in (0, 1)]
    m["videoParameters"] = dict(meta["videoParameters"], numberOfSequentialFields=len(m["fields"]))
    json.dump(m, open(sub + ".tbc.json", "w"))
    subprocess.run(["ld-dropout-correct", "-q", "--input-json", sub + ".tbc.json", "--output-json",
                    sub + "_doc.tbc.json", sub + ".tbc", sub + "_doc.tbc"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"[qc] newdecode v6 on {len(pics)} pictures around {len(keys)} key pictures")
    # one newdecode at a time across all users of the GPU: wait for the lock
    p = subprocess.Popen(["flock", "-w", str(int(lock_wait)), GPU_LOCK, MLPY, NDWRAP, nd, sub + "_doc.tbc", "-",
                          "--scratch", os.path.join(work, "nd"), "--video", "1", "--dropout_mask", "1"],
                         stdout=subprocess.PIPE, stderr=open(log_to, "w"))
    data = p.stdout.read()
    p.wait()
    if not data:
        log(f"[qc] newdecode produced nothing (GPU lock not had in {lock_wait:.0f} s?): see {log_to}")
        return {}
    hdr_end = data.index(b"\n") + 1
    hdr = data[:hdr_end].decode()
    w = int(re.search(r" W(\d+)", hdr).group(1))
    h = int(re.search(r" H(\d+)", hdr).group(1))
    fb = w * h * 3 * 2
    res, pos = {}, hdr_end
    want = set(keys)
    for pic in pics:
        pos = data.index(b"\n", pos) + 1
        if pos + fb > len(data):
            break
        if pic in want:
            y = np.frombuffer(data[pos:pos + fb], "<u2").reshape(3, h, w)
            if y.shape == (3, OH, OW):
                res[pic] = y.copy()
        pos += fb
    if cache and res:
        np.savez(cache, **{str(k): v for k, v in res.items()})
    return res


def add_ghost(ndr, ndy, tY):
    for k, y in ndy.items():
        if "comb" in ndr[k] and all(x in tY for x in (k - 1, k, k + 1)):
            ndr[k]["ghost"] = ghost_coef(y[0], tY[k], tY[k - 1], tY[k + 1])


def nd_summary(ndr, frames=None):
    S = {}
    bg = [x for v in ndr.values() for x in v.get("bar_grain", [])]
    if bg:
        S["nd_bar_grain"] = {str(v["key"]): v["bar_grain"] for v in ndr.values() if "bar_grain" in v}
        S["nd_bar_grain_mean"] = float(np.mean(bg))
        S["nd_bar_grain_max"] = float(np.max(bg))
    md = [v["map_dec"] for v in ndr.values() if "map_dec" in v]
    if md:
        S["map_v6"] = qcmap.summarise(md)
    S.update(map_err_summary(ndr.values(), "map_err_v6"))
    gd = [v["graphic_dec"] for v in ndr.values() if "graphic_dec" in v]
    if gd:
        S["graphic_v6"] = qcmap.summarise(gd)
    S["nd_flat_grain"] = _flat_grain(ndr, frames)
    cb = [v["comb"] for v in ndr.values() if "comb" in v]
    if cb:
        S["nd_comb"] = float(np.mean(cb))
    gh = [v["ghost"] for v in ndr.values() if "ghost" in v]
    if gh:
        S["nd_ghost"] = float(np.mean(gh))
    return S


def worst_bar_ratio(c, b):
    """Largest candidate / baseline grain ratio over the bars both have."""
    r = [x / y for k in c if k in b for x, y in zip(c[k], b[k]) if y > 0]
    return float(max(r)) if r else None


# --------------------------------------------------------------------------- stamps
def measure_stamp():
    """md5 of the code that produces the stored per-frame scores (qcmap, qcmetrics and quickcheck's
    scoring functions; not the gates or the report)."""
    import hashlib
    import inspect
    h = hashlib.md5()
    for f in ("qcmap.py", "qcmetrics.py"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    for fn in (Scorer, summarise, score_decoded, decoded_summary, nd_summary, dropout_error, weights_check,
               truth_file, truth_floor, _flat_grain, _gate_frame, comb_index, ghost_coef, audio_metrics, add_ghost,
               map_err_summary, truth_decoded):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()[:12]


def truth_stamp(pdir):
    """Identity of the held-run truth files (name, size, mtime of ref/truth*.npy and runmean*.npy)."""
    import hashlib
    h = hashlib.md5()
    d = os.path.join(pdir, "ref")
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if n.startswith(("truth", "runmean")) and n.endswith(".npy"):
            st = os.stat(os.path.join(d, n))
            h.update(f"{n}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def file_md5(path):
    import hashlib
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:12]


def rescore_baseline(a, bdir, base):
    """Re-score a kept baseline stack in place with the current code and panel file."""
    dec = base.get("decoder") or (os.path.dirname(a.nd_base) if a.nd_base else "t2d")
    cmd = [PY, os.path.abspath(__file__), "--stacker", base["stacker"], "--panel", a.panel, "--panel-file",
           a.panel_file, "--decoder", dec, "--from-stack", bdir, "--out", bdir, "--gpu-wait", str(a.gpu_wait)]
    log(f"[qc] baseline {bdir} was scored by other code / panel: re-scoring it from its kept stack")
    subprocess.run(cmd, stdout=open(os.path.join(bdir, "rescore.log"), "w"), stderr=subprocess.STDOUT)
    return json.load(open(os.path.join(bdir, "results.json")))


# --------------------------------------------------------------------------- report
def ratio(a, b):
    return a / b if (a is not None and b not in (None, 0)) else None


def compare(cand, base):
    """[(check, candidate, baseline, ratio, verdict)]; verdict PASS/FAIL/info."""
    rows = []

    def add(name, c, b, th, worse_if="higher", tol_abs=0.0):
        nan = lambda x: x is None or (isinstance(x, float) and not np.isfinite(x))
        if nan(c):
            return              # not measured on this panel
        b = None if nan(b) else b
        r = ratio(c, b)
        if b is None or th is None:
            v = "info"
        elif worse_if == "higher":
            v = "FAIL" if c > b * th + tol_abs else "PASS"
        else:
            v = "FAIL" if c < b * th - tol_abs else "PASS"
        rows.append((name, c, b, r, v))

    cs, bs = cand["summary"], base["summary"] if base else {}
    w = cand.get("weights")
    bw = ((base or {}).get("weights") or {})
    if w and w.get("status", "ok") != "ok":
        rows.append(("master weights: insufficient data (" + w.get("why", "") + ")", None, None, None, "n/a"))
    elif w:
        note = w.get("status_note")
        rows.append(("master weights (blend excess vs measured-noise optimum)" + (f" [{note}]" if note else ""),
                     w["max_excess"], bw.get("max_excess"), None,
                     "info" if note else ("PASS" if w["max_excess"] <= TH_W_EXCESS else "FAIL")))
        rows.append(("master weights (max weight above optimal)", w["max_dev"],
                     bw.get("max_dev"), None, "info" if note else ("PASS" if w["max_dev"] <= TH_W_DEV else "FAIL")))
    g = lambda d, k, i=None: (None if d.get(k) is None else (d[k] if i is None else d[k][i]))
    for i, bn in enumerate(qm.BAND_NAMES):
        add(f"noise flat stills {bn} MHz", g(cs, "noise_flat", i), g(bs, "noise_flat", i), TH_NOISE)
    # info only: the pre-10-26 stacker's gain fit flattened black frames (rms 35 against ~150 that
    # the captures' noise allows), so a ratio to such a baseline means nothing
    for i, bn in enumerate(qm.BAND_NAMES):
        add(f"noise black frames {bn} MHz", g(cs, "noise_black", i), g(bs, "noise_black", i), None)
    for i, bn in enumerate(qm.BAND_NAMES):
        add(f"error vs master-0-only truth {bn} MHz (sensitivity, info)", g(cs, "err_run_truth0", i),
            g(bs, "err_run_truth0", i), None)
    for i, bn in enumerate(qm.BAND_NAMES):
        add(f"error vs held-run truth {bn} MHz", g(cs, "err_run", i), g(bs, "err_run", i), TH_NOISE)
    add("detail textured stills 2-3.5 MHz", g(cs, "detail_tex", 1), g(bs, "detail_tex", 1), TH_DETAIL, "lower")
    add("detail textured stills 0-2 MHz", g(cs, "detail_tex", 0), g(bs, "detail_tex", 0), TH_DETAIL, "lower")
    if cs.get("gain_run"):
        add("detail gain held textured 2-3.5 MHz", g(cs, "gain_run", 1), g(bs, "gain_run", 1), TH_DETAIL, "lower")
    # the count depends on each stacker's flagging convention: info only; gate on the error there
    add("residual dropout samples (count, info)", cs.get("resid_do"), bs.get("resid_do"), None)
    add("error at dropout pixels vs held-run truth", cs.get("do_err"), bs.get("do_err"), TH_NOISE)
    add("flicker level (16-bit units)", cs.get("flicker_level"), bs.get("flicker_level"), TH_FLICKER, tol_abs=0.5)
    add("flicker hue (deg)", cs.get("flicker_hue_deg"), bs.get("flicker_hue_deg"), TH_FLICKER, tol_abs=0.02)
    add("flicker saturation (%)", cs.get("flicker_sat_pct"), bs.get("flicker_sat_pct"), TH_FLICKER, tol_abs=0.02)
    add("bars grain Y (mean of bars)", cs.get("bar_grain_mean"), bs.get("bar_grain_mean"), TH_GRAIN)
    add("bars grain Y (worst bar)", cs.get("bar_grain_max"), bs.get("bar_grain_max"), TH_GRAIN)
    add("decoded flat grain Y", cs.get("y_flat_grain"), bs.get("y_flat_grain"), TH_GRAIN)
    add("decoded detail Y 2-3.5 MHz", g(cs, "y_detail", 1), g(bs, "y_detail", 1), TH_DETAIL, "lower")
    add("decoded detail Y 3.5-5.5 MHz", g(cs, "y_detail", 2), g(bs, "y_detail", 2), TH_DETAIL, "lower")
    if cs.get("bar_grain") and bs.get("bar_grain"):
        wr = worst_bar_ratio(cs["bar_grain"], bs["bar_grain"])
        rows.append(("bars grain Y, worst bar cand/base (t2d)", wr, 1.0, wr,
                     "FAIL" if wr is not None and wr > TH_GRAIN else "PASS"))
    add("nd bars grain Y (newdecode, mean of bars)", cs.get("nd_bar_grain_mean"), bs.get("nd_bar_grain_mean"), TH_GRAIN)
    add("nd bars grain Y (newdecode, worst bar)", cs.get("nd_bar_grain_max"), bs.get("nd_bar_grain_max"), TH_GRAIN)
    if cs.get("nd_bar_grain") and bs.get("nd_bar_grain"):
        wr = worst_bar_ratio(cs["nd_bar_grain"], bs["nd_bar_grain"])
        rows.append(("nd bars grain Y, worst bar cand/base (newdecode)", wr, 1.0, wr,
                     "FAIL" if wr is not None and wr > TH_GRAIN else "PASS"))
    add("nd flat grain Y (newdecode)", cs.get("nd_flat_grain"), bs.get("nd_flat_grain"), TH_GRAIN)
    if isinstance(cand.get("newdecode"), dict) and base is not None and (base.get("v6_missing") or not bs.get("nd_bar_grain_mean") and cs.get("nd_bar_grain_mean")):
        rows.append(("newdecode baseline decoded", None, None, None, "FAIL"))
    # primary map gate: error vs the held-run truth in map text / line and paper areas
    for src, nm in (("map_err_comp", "stack"), ("map_err_dec", "t2d"), ("map_err_v6", "newdecode vs t2d truth")):
        cm, bm = cs.get(src) or {}, bs.get(src) or {}
        if cm:
            rows.append((f"map pictures in held runs ({nm})", cm.get("pictures"), bm.get("pictures"), None, "info"))
        for lab in ("text", "paper"):
            add(f"map {lab}-area error vs held-run truth ({nm})", cm.get(lab), bm.get(lab), TH_MAP_ERR)
    per = {"map_comp": ("frames", "map_comp"), "map_dec": ("decoded", "map_dec"), "map_v6": ("newdecode", "map_dec")}
    for src, nm in (("map_comp", "stack"), ("map_dec", "t2d"), ("map_v6", "newdecode")):
        cm, bm = cs.get(src) or {}, bs.get(src) or {}
        th_r = TH_MAP_RISE if src == "map_comp" else TH_MAP_RISE_DEC
        add(f"map line/text rise 10-90% ({nm}, samples)", cm.get("rise"), bm.get("rise"), th_r)
        add(f"map text-area 2-5 MHz detail ({nm}, info: counts noise as detail)", cm.get("text"), bm.get("text"), None)
        add(f"map paper 2-5 MHz noise ({nm})", cm.get("paper"), bm.get("paper"), None)
        # per picture: the worst map picture may not hide behind the mean
        sec, fld = per[src]
        cf_, bf_ = cand.get(sec), (base or {}).get(sec)
        if isinstance(cf_, dict) and isinstance(bf_, dict):
            rr, tr, few, ne = [], [], [], {}
            for k, v in cf_.items():
                c, b = (v or {}).get(fld), ((bf_.get(k) or {}).get(fld))
                if not c or not b or c.get("rise") is None or b.get("rise") is None:
                    continue
                n = min(c.get("n_edges") or 0, b.get("n_edges") or 0)
                ne[k] = n
                if n < MIN_PIC_EDGES:
                    few.append(k)               # too few edges: even the baseline's value is unreliable
                else:
                    rr.append(c["rise"] / b["rise"])
                if None not in (c.get("text_ms"), c.get("paper_ms"), b.get("text_ms"), b.get("paper_ms")):
                    nc = np.sqrt(max(c["text_ms"] - c["paper_ms"], 0))
                    nb_ = np.sqrt(max(b["text_ms"] - b["paper_ms"], 0))
                    if nb_ > 0:
                        tr.append(nc / nb_)
            if ne:
                rows.append((f"map pictures with >= {MIN_PIC_EDGES} edges gated ({nm}); edges per picture: "
                             + " ".join(f"{k}:{n}" for k, n in sorted(ne.items(), key=lambda x: int(x[0]))),
                             len(rr), len(ne), None, "info"))
            if rr:
                w = max(rr)
                th_p = TH_MAP_RISE_PIC if src == "map_comp" else TH_MAP_RISE_PIC_DEC
                rows.append((f"map rise, worst picture cand/base ({nm})", w, 1.0, w,
                             "FAIL" if w > th_p else "PASS"))
            if tr:
                w = min(tr)
                rows.append((f"map text detail, worst picture cand/base ({nm}, info)", w, 1.0, w, "info"))
    for src, nm in (("graphic_comp", "stack"), ("graphic_dec", "t2d"), ("graphic_v6", "newdecode")):
        cm, bm = cs.get(src) or {}, bs.get(src) or {}
        add(f"bar / test card edge rise 10-90% ({nm}, samples)", cm.get("rise"), bm.get("rise"), TH_GRAPHIC_RISE)
    add("motion combing (t2d)", cs.get("comb_t2d"), bs.get("comb_t2d"), TH_GRAIN)
    add("motion combing (newdecode)", cs.get("nd_comb"), bs.get("nd_comb"), TH_GRAIN)
    if cs.get("nd_ghost") is not None:
        bg = bs.get("nd_ghost")
        rows.append(("motion ghosting toward neighbours (newdecode, 0..1)", cs["nd_ghost"], bg, None,
                     "info" if bg is None else ("FAIL" if cs["nd_ghost"] > bg + 0.05 else "PASS")))
    ca, ba = cs.get("audio") or {}, bs.get("audio") or {}
    if ca:
        for k, nm in (("programme_muted_s", "audio: programme seconds muted by squelch"),
                      ("mono_uncorrelated_s", "audio: mono-combined seconds with uncorrelated L/R")):
            b = ba.get(k)
            rows.append((nm, ca[k], b, None, "info" if b is None else ("FAIL" if ca[k] > b else "PASS")))
        rows.append(("audio: programme / mono seconds", f"{ca['programme_s']}/{ca['mono_s']}",
                     f"{ba.get('programme_s')}/{ba.get('mono_s')}" if ba else None, None, "info"))
    add("vs production stack rms", cs.get("vs_prod_rms"), bs.get("vs_prod_rms"), None)
    return rows


def print_table(rows, cand, base):
    print(f"\nquickcheck  candidate {cand['stacker_md5'][:8]} ({cand['stacker']})"
          + (f"  vs baseline {base['stacker_md5'][:8]} ({base['stacker']})" if base else ""))
    w = cand.get("weights")
    if w and w.get("status", "ok") == "ok":
        print("  master groups: " + " | ".join(",".join(n.split("_")[-1] for n in g) for g in w["groups"]))
        for b, bn in enumerate(qm.BAND_NAMES):
            if w["w_log"][b] is None:
                print(f"  {bn:>8} MHz  not judged (masters' errors not separable here)")
                continue
            print(f"  {bn:>8} MHz  w_log " + " ".join(f"{x:.2f}" for x in w["w_log"][b])
                  + "   w_opt " + " ".join(f"{x:.2f}" for x in w["w_opt"][b])
                  + "   V " + " ".join(f"{x:.3g}" for x in w["V"][b]) + f"   excess {w['excess'][b]*100:+.0f}%")
    print(f"  {'check':56} {'cand':>10} {'base':>10} {'ratio':>7}  verdict")
    for name, c, b, r, v in rows:
        fmt = lambda x: "-" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))
        print(f"  {name:56} {fmt(c):>10} {fmt(b):>10} {('-' if r is None else f'{r:.3f}'):>7}  {v}")
    nf = sum(v == "FAIL" for *_, v in rows)
    print(f"  => {'FAIL (' + str(nf) + ')' if nf else 'PASS'}")
    return nf


def contact_sheet(out_png, crops, base_crops, labels):
    from PIL import Image, ImageDraw
    rows = []
    for lab in labels:
        c = crops.get(lab)
        b = base_crops.get(lab) if base_crops is not None else None
        if c is None:
            continue
        if b is None or b.shape != c.shape:
            rows.append((lab, c))
            continue
        d = np.clip(np.abs(c.astype(np.int16) - b.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        rows.append((lab, np.concatenate([c, np.full((c.shape[0], 4, 3), 255, np.uint8), b,
                                          np.full((c.shape[0], 4, 3), 255, np.uint8), d], 1)))
    if not rows:
        return
    wmax = max(r.shape[1] for _, r in rows)
    pad = 14
    H = sum(r.shape[0] + pad for _, r in rows) + pad
    im = Image.new("RGB", (wmax, H), (40, 40, 40))
    dr = ImageDraw.Draw(im)
    y = pad
    dr.text((2, 0), "candidate | baseline | |diff| x4", fill=(255, 255, 0))
    for lab, r in rows:
        dr.text((2, y - 12), lab, fill=(255, 255, 255))
        im.paste(Image.fromarray(r), (0, y))
        y += r.shape[0] + pad
    im.save(out_png)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stacker", required=True)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--panel-file", default=os.environ.get("QC_PANEL_FILE", "panel.json"),
                    help="panel description in the panel dir (panel_v2.json: raw-capture truth)")
    ap.add_argument("--decoder", default="t2d",
                    help="t2d | none | v6 (= %s) | a newdecode.py, or a directory holding one" % ND)
    ap.add_argument("--baseline-decoder", default=None,
                    help="newdecode for the baseline's v6 decode (default: the candidate's)")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=12,
                    help="stacker workers (at most 12; fewer when RAM is short: ~0.4 GB each)")
    ap.add_argument("--block", type=int, default=10,
                    help="LDSTACK_BLOCK: frames per worker block (production 24: 320 pictures would be 14 blocks; "
                         "10 gives 32 blocks; keep it equal for candidate and baseline)")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--gpu-wait", type=float, default=240, help="v6: minutes to wait for aiv-gpu.lock before skipping")
    ap.add_argument("--nd-context", type=int, default=8,
                    help="v6: each key picture is decoded inside its run of consecutive panel pictures, up to "
                         "this many either side (newdecode's held-still averaging needs the neighbours)")
    ap.add_argument("--stack-extra", default="", help="extra stacker arguments")
    ap.add_argument("--keep", action="store_true", help="keep the stacked panel TBC in the results dir")
    ap.add_argument("--allow-stale", action="store_true",
                    help="compare against a baseline scored by other code / panel (normally refused)")
    ap.add_argument("--from-stack", default=None, metavar="RESULTS",
                    help="rescore the stack kept (--keep) in RESULTS instead of stacking (metric changes)")
    a = ap.parse_args()
    a.panel = os.path.abspath(a.panel)
    a.nd = nd_script(a.decoder)
    a.nd_base = nd_script(a.baseline_decoder) if a.baseline_decoder else a.nd
    T0 = time.time()
    tm = {}
    panel = json.load(open(os.path.join(a.panel, a.panel_file)))
    panel["_dir"] = a.panel
    pkgtop, mod, h = make_pkg(a.stacker, os.path.join(a.panel, "pkg"))
    out = a.out or os.path.join(a.panel, "results", f"{h[:8]}_{time.strftime('%m%d-%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    shm = "/dev/shm" if shutil.disk_usage("/dev/shm").free > 4e9 else a.panel
    work = tempfile.mkdtemp(prefix="qc_", dir=shm)
    base = json.load(open(os.path.join(a.baseline, "results.json"))) if a.baseline else None
    stamp = {"measure": measure_stamp(), "panel_file": a.panel_file,
             "panel": file_md5(os.path.join(a.panel, a.panel_file)), "truth": truth_stamp(a.panel)}
    if base is not None and base.get("stamp", {}).get("measure") != stamp["measure"] \
            or base is not None and base.get("stamp", {}).get("panel") != stamp["panel"] \
            or base is not None and base.get("stamp", {}).get("truth") != stamp["truth"]:
        if os.path.exists(os.path.join(a.baseline, "stk.tbc")) or panel.get("no_stack"):
            base = rescore_baseline(a, a.baseline, base)
        if base.get("stamp", {}).get("measure") != stamp["measure"] and not a.allow_stale:
            raise SystemExit(f"[qc] baseline {a.baseline} is stamped {base.get('stamp')} but this code / panel is "
                             f"{stamp}: re-score it (keep its stack) or pass --allow-stale")
    if base is not None:
        base["summary"] = resummarise(panel, base)
        if base.get("log"):
            base["weights"] = weights_check(panel, base["log"])   # judged by this panel's noise measurement
    try:
        caps = [os.path.join(a.panel, "caps", n + ".tbc") for n in panel["caps"]]
        stk = os.path.join(work, "stk")
        env = dict(os.environ, PYTHONPATH=pkgtop, LDSTACK_FRAME_STATS=os.path.join(work, "frames.tsv"),
                   LDSTACK_GAP_MAX="0", LDSTACK_BLOCK=str(a.block))
        # RAM: ~0.35 GB PSS per worker on the 12-capture panel; keep MemAvailable above ~6 GB
        avail = int(next(ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemAvailable"))) / 1e6
        cap = max(2, min(12, int((avail - 6) / 0.4)))
        if a.jobs > cap:
            log(f"[qc] {avail:.1f} GB available: stacker limited to {cap} workers")
            a.jobs = cap
        cmd = [PY, "-m", mod, "--PAL", "--jobs", str(a.jobs), "--cross_master"] + panel.get("stack_args", []) \
            + a.stack_extra.split() \
            + ["-o", stk] + caps
        log(f"[qc] stacking {len(panel['stack_keys'])} pictures x {len(caps)} captures with {a.stacker} ({h[:8]})")
        t = time.time()
        if panel.get("no_stack"):
            # a single-copy side: nothing to stack, the capture is the stack (decoder checks only)
            for ext in (".tbc", ".tbc.json", ".tbc.db", ".pcm"):
                if os.path.exists(caps[0][:-4] + ext):
                    os.symlink(caps[0][:-4] + ext, stk + ext)
            open(os.path.join(out, "stack.log"), "w").write("single capture: no stack\n")
            lf = open(os.devnull, "w")
            proc = _Done()
        elif a.from_stack:
            # rescore a kept stack (--keep): no stacking
            for ext in (".tbc", ".tbc.json", ".tbc.db", ".pcm"):
                if os.path.exists(os.path.join(a.from_stack, "stk" + ext)):
                    os.symlink(os.path.abspath(os.path.join(a.from_stack, "stk" + ext)), stk + ext)
            if os.path.abspath(a.from_stack) != os.path.abspath(out):
                shutil.copy(os.path.join(a.from_stack, "stack.log"), os.path.join(out, "stack.log"))
                # the rescored dir can serve as a baseline: link the stack and the decode caches
                for n in os.listdir(a.from_stack):
                    if n.startswith("stk.") or n.startswith("v6cache_") or n == "audio.json":
                        dst = os.path.join(out, n)
                        if not os.path.lexists(dst):
                            os.symlink(os.path.abspath(os.path.join(a.from_stack, n)), dst)
            lf = open(os.devnull, "w")
            proc = _Done()
        else:
            lf = open(os.path.join(out, "stack.log"), "w")
            proc = subprocess.Popen(cmd, cwd=pkgtop, env=env, stdout=lf, stderr=subprocess.STDOUT)
        peak = [0]
        threading.Thread(target=watch_rss, args=(proc, peak), daemon=True).start()
        sc = Scorer(panel, a.panel)
        keys = panel["stack_keys"]
        frames = {}
        pool = cf.ThreadPoolExecutor(1 if a.sequential else 8)
        futs = []
        if a.sequential:
            proc.wait()
        # score each frame as soon as the stacker has written it (output is in picture order)
        fd = None
        i = 0
        while i < len(keys):
            if fd is None:
                if os.path.exists(stk + ".tbc"):
                    fd = os.open(stk + ".tbc", os.O_RDONLY)
                elif proc.poll() is not None:
                    break
                else:
                    time.sleep(0.2)
                    continue
            if os.fstat(fd).st_size >= (i + 1) * 2 * FBYTES:
                buf = os.pread(fd, 2 * FBYTES, i * 2 * FBYTES)
                fr = np.frombuffer(buf, "<u2").reshape(2, qm.H, qm.W)
                futs.append((i, keys[i], pool.submit(sc.frame, keys[i], fr)))
                i += 1
            elif proc.poll() is not None and os.fstat(fd).st_size < (i + 1) * 2 * FBYTES:
                break
            else:
                time.sleep(0.1)
        rc = proc.wait()
        lf.close()
        tm["stack"] = time.time() - t
        res_mem = peak[0]
        text = open(os.path.join(out, "stack.log")).read()
        if rc != 0 or not os.path.exists(stk + ".tbc.json"):
            log(text[-2000:])
            raise SystemExit(f"[qc] stacker failed (rc {rc})")
        meta = json.load(open(stk + ".tbc.json"))
        flds = meta["fields"]
        got = qm.frame_keys(flds, panel.get("clv", False))
        t = time.time()
        bad = [(j, k) for j, k, _ in futs if j >= len(got) or got[j] != k]
        if bad or len(got) != len(keys):
            # the stacker's frame list differs from the panel's (padding or a changed sieve): rescore by picture
            log(f"[qc] output order differs from the panel ({len(got)} frames, {len(bad)} mismatched): rescoring by picture")
            futs = []
            with open(stk + ".tbc", "rb") as f:
                seen = set()
                for j, k in enumerate(got):
                    if k is None or k in seen:
                        continue
                    seen.add(k)
                    f.seek(j * 2 * FBYTES)
                    fr = np.frombuffer(f.read(2 * FBYTES), "<u2").reshape(2, qm.H, qm.W)
                    futs.append((j, k, pool.submit(sc.frame, k, fr)))
        for j, k, fu in futs:
            frames[k] = fu.result()
        pool.shutdown()
        tm["score_tail"] = time.time() - t
        k2out = {}
        for j, k in enumerate(got):
            if k is not None and k not in k2out:
                k2out[k] = j
        # residual dropouts (samples still flagged in the stacked output)
        rdo, rdo_fr = 0, 0
        for j, k in enumerate(got):
            n = 0
            for fl in flds[2 * j:2 * j + 2]:
                do = fl.get("dropOuts") or {}
                n += sum(e - s for s, e in zip(do.get("startx", []), do.get("endx", [])))
            rdo += n
            rdo_fr += n > 0
        lg = parse_log(text)
        res = {"stacker": os.path.abspath(a.stacker), "stacker_md5": h, "panel": os.path.abspath(a.panel),
               "log": lg, "frames": {str(k): v for k, v in frames.items()}}
        S = summarise(panel, frames, {"resid_do": int(rdo), "resid_do_frames": int(rdo_fr),
                                      "out_frames": len(got)})
        res["weights"] = weights_check(panel, lg)
        # ---- decode
        crops = {}
        mset, tY = set(), {}
        if a.decoder != "none" and panel.get("decode"):
            t = time.time()
            d = panel["decode"]
            want = {int(k) for k in list(d.get("bar_regions", {})) + list(d.get("tex_blocks", {}))}
            want |= {int(c["pic"]) for c in d.get("crops", [])}
            want |= {int(k) for k in panel.get("map", {})}
            want |= {int(k) for k in panel.get("graphic", {})}
            want |= {k + d for k in panel.get("motion", []) for d in (-1, 0, 1)}
            idx = {k2out[k]: k for k in want if k in k2out}
            dec = decode_ranges(stk + ".tbc", idx, a.sequential)
            dres = {}
            for j, yuv in dec.items():
                k = idx[j]
                dres[k] = score_decoded(panel, k, yuv)
                for c in d.get("crops", []):
                    if int(c["pic"]) == k:
                        crops[c["label"]] = crop_rgb(yuv, c)
            S.update(decoded_summary(dres, frames))
            res["decoded"] = {str(k): v for k, v in dres.items()}
            mset = {k + d for k in panel.get("motion", []) for d in (-1, 0, 1)}
            tY = {idx[j]: y[0] for j, y in dec.items() if idx[j] in mset}
            tm["decode_t2d"] = time.time() - t
        base_nd_crops = {}
        if a.nd and panel.get("decode", {}).get("nd_frames"):
            if gpu_free():
                t = time.time()
                # cached beside the stack: a --from-stack rescore does not decode again
                ctx = panel.get("nd_context", a.nd_context)
                ndy = v6_decode(panel, stk, meta, os.path.join(work, "nd_c"), os.path.join(out, "nd.log"), ctx,
                                cache_dir=a.from_stack or out, lock_wait=a.gpu_wait * 60, nd=a.nd)
                ndr = {k: score_decoded(panel, k, y) for k, y in ndy.items()}
                add_ghost(ndr, ndy, tY if a.decoder != "none" else {})
                S.update(nd_summary(ndr, frames))
                for k, y in ndy.items():
                    for c in panel["decode"].get("crops", []):
                        if int(c["pic"]) == k:
                            crops["nd " + c["label"]] = crop_rgb(y, c)
                res["newdecode"] = {str(k): v for k, v in ndr.items()}
                tm["decode_v6"] = time.time() - t
                # the baseline through the same newdecode, same pictures, same context (cached)
                if base is not None:
                    t = time.time()
                    bstk = os.path.join(a.baseline, "stk")
                    if panel.get("no_stack") and not os.path.exists(bstk + ".tbc"):
                        bstk = caps[0][:-4]              # one copy: the capture is the baseline's "stack" too
                    if not os.path.exists(bstk + ".tbc"):
                        log("[qc] baseline has no kept stack (run it with --keep): v6 baseline missing")
                        base["v6_missing"] = True
                    else:
                        bmeta = json.load(open(bstk + ".tbc.json"))
                        bny = v6_decode(panel, bstk, bmeta, os.path.join(work, "nd_b"),
                                        os.path.join(out, "nd_baseline.log"), ctx, cache_dir=a.baseline,
                                        lock_wait=a.gpu_wait * 60, nd=a.nd_base)
                        bndr = {k: score_decoded(panel, k, y) for k, y in bny.items()}
                        # the baseline's own transform2d motion frames for its ghost measure
                        bk2o = {k: j for j, k in enumerate(qm.frame_keys(bmeta["fields"], panel.get("clv", False)))
                                if k is not None}
                        bm = {bk2o[k]: k for k in mset if k in bk2o}
                        bY = {bm[j]: y[0] for j, y in decode_ranges(bstk + ".tbc", bm, a.sequential).items()}
                        add_ghost(bndr, bny, bY)
                        for key in [k for k in base["summary"] if k.startswith("nd_") or k == "map_v6"]:
                            del base["summary"][key]
                        base["summary"].update(nd_summary(bndr, base.get("frames")))
                        base["newdecode"] = {str(k): v for k, v in bndr.items()}
                        res["baseline_v6"] = {"summary": nd_summary(bndr, base.get("frames")),
                                              "frames": {str(k): v for k, v in bndr.items()}}
                        for k, y in bny.items():
                            for c in panel["decode"].get("crops", []):
                                if int(c["pic"]) == k:
                                    base_nd_crops["nd " + c["label"]] = crop_rgb(y, c)
                    tm["decode_v6_baseline"] = time.time() - t
            else:
                log("[qc] production newdecode (nd_gpuwrap) running: newdecode stage skipped")
                res["newdecode"] = "skipped: GPU busy"
        t = time.time()
        sts = {"cand": stk}
        if base is not None and a.baseline and os.path.exists(os.path.join(a.baseline, "stk.tbc")):
            sts["base"] = os.path.join(a.baseline, "stk")
        de = dropout_error(panel, a.panel, sts)
        S["do_err"] = (de.get("cand") or {}).get("rms")
        S["do_err_pixels"] = (de.get("cand") or {}).get("pixels")
        if base is not None and de.get("base"):
            base["summary"]["do_err"] = de["base"]["rms"]
            base["summary"]["do_err_pixels"] = de["base"]["pixels"]
        tm["dropout_err"] = time.time() - t
        t = time.time()
        S["audio"] = audio_metrics(stk, work)
        if base is not None and a.baseline:
            ac = os.path.join(a.baseline, "audio.json")
            if os.path.exists(ac):
                base["summary"]["audio"] = json.load(open(ac))
            elif os.path.exists(os.path.join(a.baseline, "stk.pcm")):
                base["summary"]["audio"] = audio_metrics(os.path.join(a.baseline, "stk"), work)
                json.dump(base["summary"]["audio"], open(ac, "w"))
        tm["audio"] = time.time() - t
        res["summary"] = S
        res["stamp"] = stamp
        res["decoder"] = os.path.dirname(a.nd) if a.nd else a.decoder
        res["stack_peak_rss_gb"] = res_mem / 1e9
        res["stack_settings"] = {"jobs": a.jobs, "block": a.block, "extra": a.stack_extra}
        tm["total"] = time.time() - T0
        res["timing"] = tm
        np.savez_compressed(os.path.join(out, "crops.npz"), **crops)
        base_crops = None
        if a.baseline and os.path.exists(os.path.join(a.baseline, "crops.npz")):
            base_crops = dict(np.load(os.path.join(a.baseline, "crops.npz")))
        if base_nd_crops:
            base_crops = dict(base_crops or {}, **base_nd_crops)
        labels = [c["label"] for c in panel.get("decode", {}).get("crops", [])]
        labels += ["nd " + l for l in labels]
        contact_sheet(os.path.join(out, "contact.png"), crops, base_crops, labels)
        rows = compare(res, base)
        res["checks"] = [list(r) for r in rows]
        json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
        if a.keep and not a.from_stack:
            for ext in (".tbc", ".tbc.json", ".tbc.db", ".pcm"):
                if os.path.exists(stk + ext):
                    shutil.copy(stk + ext, os.path.join(out, "stk" + ext))
        nf = print_table(rows, res, base)
        if a.nd and not (isinstance(res.get("newdecode"), dict) and res["newdecode"]):
            print("  !! newdecode stage SKIPPED or empty: v6 checks not run, exit status 2")
        print("  timing: " + ", ".join(f"{k} {v:.0f}s" for k, v in tm.items())
              + f"; stacker peak PSS {res['stack_peak_rss_gb']:.1f} GB ({a.jobs} workers, blocks of {a.block})")
        print(f"  results: {out}")
        if nf:
            return 1
        return 2 if a.nd and not (isinstance(res.get("newdecode"), dict) and res["newdecode"]) else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
