# SPDX-License-Identifier: GPL-3.0-or-later
"""A panel from short capture decodes of one stretch of a side (e.g. the e2e South s3000 segment:
`ld-decode --seek` decodes of every capture), stacked whole.

  make_segment_panel.py <spec.json> <panel dir>

spec.json: {"side", "caps": [capture .tbc paths], "reference": {"tbc": <a stack of the same captures>},
            "segments": [[p0, p1, role], ...], "runs": [[r0, r1, name, [scored pictures]], ...],
            "crop_spec": [[picture, label, x, y, w, h], ...], "nd_spec": [newdecode key pictures]}
The captures are linked into <panel dir>/caps/ (no copy).  Held-run references come from the
reference stack's frames of each run that are not scored.  Then run prepare_panel.py and
measure_masters.py on the panel dir as for North.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcmetrics as qm               # noqa: E402
from extract_panel import extract_ref, ref_pic2frame   # noqa: E402


def main():
    spec = json.load(open(sys.argv[1]))
    pdir = sys.argv[2]
    os.makedirs(os.path.join(pdir, "caps"), exist_ok=True)
    names = []
    for t in spec["caps"]:
        b = t[:-4]
        n = os.path.basename(b)
        names.append(n)
        for ext in (".tbc", ".tbc.json", ".tbc.db", ".pcm"):
            dst = os.path.join(pdir, "caps", n + ext)
            if os.path.exists(b + ext) and not os.path.lexists(dst):
                os.symlink(os.path.abspath(b + ext), dst)
    # the pictures the reference stack has (a capture's raw VBI can carry stray misreads that the
    # stacker's sieve drops; prepare_panel.py checks the list against the stacker's own index)
    keys = set(ref_pic2frame(spec["reference"]))
    panel = {"side": spec["side"], "caps": names, "stack_keys": sorted(keys),
             "segments": [{"p0": a, "p1": b, "role": r} for a, b, r in spec["segments"]],
             "runs": [{"r0": a, "r1": b, "name": n, "stacked": st, "textured": n == "testcard"}
                      for a, b, n, st in spec["runs"]],
             "reference": spec["reference"], "crop_spec": spec.get("crop_spec", [])}
    if spec.get("nd_spec"):
        panel["nd_spec"] = spec["nd_spec"]
    extract_ref(panel, pdir, qm.H * qm.W * 2, qm.H, qm.W)
    json.dump(panel, open(os.path.join(pdir, "panel.json"), "w"), indent=1)
    print(f"{len(names)} captures, {len(keys)} pictures; runs: "
          + ", ".join(f"{r['name']} ref {r['ref_frames']}" for r in panel["runs"]))


if __name__ == "__main__":
    main()
