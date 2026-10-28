# SPDX-License-Identifier: GPL-3.0-or-later
"""Noise invariance of the map / edge metrics (qcmap): the robust erf-fit rise width, the old monotone
walk, and the text-area 2-5 MHz detail, on clean edges with added white noise at 0.5x, 1x, 2x a typical
stack residual.

  test_rise_noise.py [panel dir with panel_v2.json truth] ...

Real edges: the held-run truths (raw-capture means, noise ~1/sqrt(n) of a stack) of map / text runs,
composite luma path and transform2d path (noise added to the composite before decoding).  Synthetic:
blurred steps / bars / letter-like strokes of known width.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcmap                    # noqa: E402
import qcmetrics as qm          # noqa: E402
import qcrender                 # noqa: E402

RESID = 420.0                   # typical stack residual rms (16-bit composite, all bands)
LEVELS = (0.0, 0.5, 1.0, 2.0, "J", "J+1")    # noise x residual; J: +-0.7 px random per-line shift (+1x noise)


def jitter(img, rng, amp=0.7):
    """Each line of each field shifted by a random +-amp samples (per-line timing jitter)."""
    out = np.empty_like(img, dtype=np.float64)
    k = np.fft.rfftfreq(img.shape[-1])
    for f in range(img.shape[0]):
        d = rng.uniform(-amp, amp, img.shape[1])
        F = np.fft.rfft(img[f].astype(np.float64), axis=-1)
        out[f] = np.fft.irfft(F * np.exp(-2j * np.pi * k[None, :] * d[:, None]), img.shape[-1], axis=-1)
    return out


def perturb(tr, lv, rng):
    if lv == "J":
        return jitter(tr, rng)
    if lv == "J+1":
        return jitter(tr, rng) + rng.normal(0, RESID, tr.shape)
    return tr + rng.normal(0, lv * RESID, tr.shape)


def synth(sigma, rng):
    """A field of luma steps / bars / strokes on paper, Gaussian-blurred by sigma samples."""
    img = np.full((qm.H, qm.W), 42000.0)
    for y0 in range(30, 300, 40):
        for x0 in range(200, 1080, 90):
            w = rng.integers(3, 40)
            img[y0:y0 + 30, x0:x0 + w] = 20000 + rng.integers(0, 8000)
    k = np.arange(-15, 16)
    g = np.exp(-0.5 * (k / sigma) ** 2)
    g /= g.sum()
    return np.apply_along_axis(lambda r: np.convolve(r, g, "same"), 1, img)


def measure(img, spec):
    fr = np.stack([img, img])
    return qcmap.measure_comp(fr, spec)


def main():
    rng = np.random.default_rng(1)
    print("synthetic (true 10-90 width = 2.563 sigma), composite-luma path")
    print("  sigma  true   level  robust   walk")
    for sigma in (0.8, 1.8):
        img = synth(sigma, rng)
        ref = np.stack([img, img])
        spec = qcmap.select(ref, np.zeros((qcrender.OH, qcrender.OW)) + 40000, seed=0)
        for lv in LEVELS:
            m = measure(perturb(np.stack([img, img]), lv, rng)[0], spec)
            print(f"  {sigma:4.1f}  {2.563 * sigma:5.2f}  {str(lv):>4}  {m['rise']:6.3f}  {m['rise_walk']:6.3f}"
                  f"   ({m['n_edges']} steps)")
    for pdir in sys.argv[1:]:
        p = json.load(open(os.path.join(pdir, "panel_v2.json")))
        print(f"\n{pdir}: real map / text edges in the held-run truths (composite luma; transform2d)")
        for r in p["runs"]:
            t = (p.get("truth") or {}).get("runs", {}).get(str(r["r0"]))
            if not t:
                continue
            P = sorted(t["floor"])[0]
            tr = np.load(os.path.join(pdir, "ref", f"truth_{r['r0']}_{P}.npy")).astype(np.float64)
            spec = qcmap.select(tr, np.zeros((qcrender.OH, qcrender.OW)) + 40000, seed=0)
            if len(spec["comp"]["steps"]) < 5:
                continue
            res = {}
            for lv in LEVELS:
                res[lv] = qcmap.measure_comp(perturb(tr, lv, rng), spec)
            # decoded path
            meta = qcrender.load_meta(p["reference"]["tbc"])
            from extract_panel import ref_pic2frame
            p2f = ref_pic2frame(p["reference"])
            k = r["stacked"][0]
            dec = {}
            with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
                frs = [np.clip(perturb(tr, lv, rng), 0, 65535).astype(np.uint16) for lv in LEVELS]
                qcrender.frames_tbc(None, meta, [p2f[k]] * len(LEVELS), os.path.join(td, "r"), frames=np.stack(frs))
                ys = qcrender.decode(os.path.join(td, "r"))
            dspec = qcmap.select(tr, ys[0][0], seed=0)
            for lv, y in zip(LEVELS, ys):
                dec[lv] = qcmap.measure_dec(y[0], dspec)
            b0, d0 = res[0.0], dec[0.0]
            net = lambda m: np.sqrt(max(m["text_ms"] - m["paper_ms"], 0)) if m["text_ms"] else float("nan")
            print(f"  run {r['name']}@{r['r0']} ({len(spec['comp']['steps'])} comp / {len(dspec['dec']['steps'])} t2d steps)")
            for lv in LEVELS:
                a, d = res[lv], dec[lv]
                print(f"    {str(lv):>4}: robust comp {a['rise']:.3f} ({100 * (a['rise'] / b0['rise'] - 1):+.1f}%)"
                      f"  t2d {d['rise']:.3f} ({100 * (d['rise'] / d0['rise'] - 1):+.1f}%)"
                      f" | walk comp {a['rise_walk']:.3f} ({100 * (a['rise_walk'] / b0['rise_walk'] - 1):+.1f}%)"
                      f"  t2d {d['rise_walk']:.3f} ({100 * (d['rise_walk'] / d0['rise_walk'] - 1):+.1f}%)"
                      f" | text detail comp {net(a):.0f} ({100 * (net(a) / net(b0) - 1):+.1f}%)"
                      f"  t2d {net(d):.0f} ({100 * (net(d) / net(d0) - 1):+.1f}%)")


if __name__ == "__main__":
    main()
