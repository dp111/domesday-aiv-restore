# SPDX-License-Identifier: GPL-3.0-or-later
"""Content-offset check: is every capture's picture numbering the same as the others'?

  check_offsets.py <panel dir> [--fix] [--max 2]

For pictures where the content changes from one picture to the next (so a one-picture slip shows),
each capture's frame k is correlated (low-passed luma thumbnails, both fields) with the median of the
OTHER captures' frames k+d, d = -max..+max.  A capture whose best d is not 0 on most such pictures (and
clearly better than d = 0) is misnumbered: its picture k is the others' k+d.  Countryside_S1 ds3: d = +1
(0.999 against 0.631 at d = 0).
--fix renumbers that capture's mini capture (VBI picture numbers in .tbc.json and .tbc.db, CAV only) by
+d, writing the originals to <capture>.tbc.json.orig / .tbc.db.orig.
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcmetrics as qm          # noqa: E402


def thumb(fr):
    a = fr[:, qm.RA, qm.CA].astype(np.float32)[:, :280, :920].reshape(2, 35, 8, 115, 8).mean(axis=(2, 4))
    return (a - a.mean()).ravel()


def corr(a, b):
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def bcd(v, n):
    return int("%0*d" % (n, v), 16)


def renumber_vbi(words, d):
    out = []
    for w in words:
        h = "%06x" % w
        if h[0] == "f" and all(c in "0123456789" for c in h[1:]):
            w = 0xF00000 | bcd(int(h[1:]) + d, 5)
        out.append(w)
    return out


def fix(base, d):
    for ext in (".tbc.json", ".tbc.db"):
        if not os.path.exists(base + ext + ".orig"):
            shutil.copy2(base + ext, base + ext + ".orig")
    m = json.load(open(base + ".tbc.json.orig"))
    for f in m["fields"]:
        v = (f.get("vbi") or {}).get("vbiData")
        if v:
            f["vbi"]["vbiData"] = renumber_vbi(v, d)
    json.dump(m, open(base + ".tbc.json", "w"))
    shutil.copy2(base + ".tbc.db.orig", base + ".tbc.db")
    db = sqlite3.connect(base + ".tbc.db")
    rows = db.execute("SELECT capture_id, field_id, vbi0, vbi1, vbi2 FROM vbi").fetchall()
    db.executemany("UPDATE vbi SET vbi0=?, vbi1=?, vbi2=? WHERE capture_id=? AND field_id=?",
                   [tuple(renumber_vbi([a, b, c], d)) + (ci, fi) for ci, fi, a, b, c in rows])
    db.commit()
    db.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--max", type=int, default=2)
    ap.add_argument("--panel-file", default="panel.json")
    a = ap.parse_args()
    p = json.load(open(os.path.join(a.panel, a.panel_file)))
    clv = p.get("clv", False)
    caps = {}
    for n in p["caps"]:
        b = os.path.join(a.panel, "caps", n)
        f = json.load(open(b + ".tbc.json"))["fields"]
        idx = {}
        for j, k in enumerate(qm.frame_keys(f, clv)):
            if k is not None and k not in idx:
                idx[k] = j
        caps[n] = (idx, np.memmap(b + ".tbc", "<u2", "r").reshape(-1, 2, qm.H, qm.W))
    keys = sorted(set.union(*[set(c[0]) for c in caps.values()]))
    th = {n: {k: thumb(c[1][c[0][k]]) for k in keys if k in c[0]} for n, c in caps.items()}
    D = range(-a.max, a.max + 1)
    report = {}
    for n in caps:
        votes, gain = [], []
        for k in keys:
            if k not in th[n]:
                continue
            ref = {}
            for d in D:
                o = [th[m][k + d] for m in caps if m != n and k + d in th[m]]
                if len(o) >= 1:
                    ref[d] = np.median(o, axis=0)
            if 0 not in ref or len(ref) < 3:
                continue
            # only where the picture changes from k-1 to k+1 (a held still cannot show a slip)
            if -1 in ref and 1 in ref and corr(ref[-1], ref[1]) > 0.98:
                continue
            c = {d: corr(th[n][k], r) for d, r in ref.items()}
            best = max(c, key=c.get)
            votes.append(best)
            gain.append(c[best] - c[0])
        if not votes:
            report[n] = {"offset": 0, "pictures": 0}
            continue
        v = np.bincount(np.array(votes) + a.max, minlength=len(D))
        best = int(np.argmax(v)) - a.max
        frac = float(v.max() / len(votes))
        report[n] = {"offset": best if best != 0 and frac > 0.6 else 0, "vote": best, "agree": round(frac, 2),
                     "pictures": len(votes), "corr_gain": round(float(np.median(gain)), 3)}
    bad = {n: r for n, r in report.items() if r["offset"]}
    print(f"{a.panel}: " + ("misnumbered: " + ", ".join(f"{n} {r['offset']:+d} ({r['agree']:.0%} of {r['pictures']} "
                                                            f"changing pictures, corr +{r['corr_gain']})"
                                                            for n, r in bad.items()) if bad else
                            f"all {len(report)} captures consistent ({min(r['pictures'] for r in report.values())}+ changing pictures each)"))
    if a.fix:
        for n, r in bad.items():
            if clv:
                print(f"  {n}: CLV renumbering not supported, fix by hand")
                continue
            fix(os.path.join(a.panel, "caps", n), r["offset"])
            print(f"  {n}: renumbered {r['offset']:+d} (originals kept as .orig)")
    json.dump(report, open(os.path.join(a.panel, "offsets.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
