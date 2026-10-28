# SPDX-License-Identifier: GPL-3.0-or-later
"""Choose a generalisation panel's segments for one side from a sparse scan of an existing stack.

  gen_select.py <scan.npz (scan_side.py --step N)> <stack.tbc> <out spec.json> [--clv] [--side NAME]

From the scan (one stripe of every Nth frame: level, contrast, chroma, detail, change between samples)
and the stack's metadata (picture keys, residual dropouts per frame):
  held   the longest held run with picture detail (>= 4 identical samples), 40 pictures, 6 scored
  held2  the longest held run with colour elsewhere, 30 pictures
  motion the longest stretch of real motion (moderate change every sample), 30 pictures
  text / saturated / dark / dropouts   8 pictures each at the most detailed bright frame (maps, text),
         the most saturated, the darkest frame with detail, and the frame with most residual dropouts
Prints the choice; quickcheck's gen_build.py decodes and assembles the panel from the spec.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcmetrics as qm          # noqa: E402
from qcrender import load_meta  # noqa: E402


def runs_of(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("stack")
    ap.add_argument("out")
    ap.add_argument("--clv", action="store_true")
    ap.add_argument("--side", default="")
    a = ap.parse_args()
    z = np.load(a.scan)
    fr, st, th = z["frames"], z["stats"][:, 0], z["thumbs"][:, 0].astype(np.float32)
    mean, sd, ch, hf = st.T
    d = np.r_[1e9, np.sqrt(((th[1:] - th[:-1]) ** 2).mean(axis=(1, 2)))]
    meta = load_meta(a.stack)
    keys = qm.frame_keys(meta["fields"], a.clv)
    flds = meta["fields"]
    rdo = np.zeros(len(keys))
    for j in range(len(keys)):
        for f in flds[2 * j:2 * j + 2]:
            do = f.get("dropOuts") or {}
            rdo[j] = rdo[j] + sum(e - s for s, e in zip(do.get("startx", []), do.get("endx", [])))
    key = lambda i: keys[int(fr[i])] if int(fr[i]) < len(keys) else None
    nb = mean > 17500
    used = []

    def free(k, n):
        return k is not None and all(abs(k - u) > n + 60 for u in used)

    segs, runs = [], []
    # held runs: identical samples (change within noise) with detail / with colour
    same = (d < np.maximum(350, 0.04 * sd)) & nb
    cand = sorted(runs_of(same), key=lambda r: -(r[1] - r[0]))
    for want, n, nm in (("detail", 40, "held"), ("colour", 30, "held2")):
        for i0, i1 in cand:
            if i1 - i0 < 3:
                break
            i0 = max(i0 - 1, 0)                 # the run starts at the sample before the first 'same'
            k0, k1 = key(i0), key(i1 - 1)
            if k0 is None or k1 is None or k1 - k0 < n + 6:
                continue
            ok = sd[i0:i1].mean() > 2500 if want == "detail" else ch[i0:i1].mean() > 2500
            mid = (k0 + k1) // 2
            if ok and free(mid, n):
                p0 = mid - n // 2
                segs.append([p0, n, nm])
                runs.append([k0 + 2, k1 - 2, nm, list(range(p0 + n // 2 - 3, p0 + n // 2 + 3))])
                used.append(p0)
                break
    # motion: moderate change at every sample for >= 4 samples
    rel = d / np.maximum(sd, 300)
    mov = (d > 400) & (rel < 0.35) & nb
    for i0, i1 in sorted(runs_of(mov), key=lambda r: -(r[1] - r[0])):
        k = key(i0)
        if i1 - i0 >= 2 and free(k, 30):
            segs.append([k, 30, "motion"])
            used.append(k)
            break
    # single pictures: map / text, saturated, dark with detail, residual dropouts
    for nm, score, m in (("text", hf, nb & (mean > 30000)), ("saturated", ch, nb),
                         ("dark", hf, nb & (mean < 22000)), ("dropouts", None, None)):
        if nm == "dropouts":
            ks = [keys[j] for j in np.argsort(-rdo)[:400]]
        else:
            ks = [key(i) for i in np.argsort(-np.where(m, score, -1e18))[:400]]
        for k in ks:
            if free(k, 8):
                segs.append([k - 3, 8, nm])
                used.append(k - 3)
                break
    segs.sort()
    spec = {"side": a.side or os.path.basename(a.stack), "clv": a.clv, "segments": segs, "runs": runs,
            "source_stack": a.stack, "motion": [[s[0], s[0] + s[1] - 1] for s in segs if s[2] == "motion"]}
    json.dump(spec, open(a.out, "w"), indent=1)
    for s in segs:
        print(s)
    print("runs", runs)


if __name__ == "__main__":
    main()
