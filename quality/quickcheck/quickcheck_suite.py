#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""quickcheck over the whole generalisation suite: every panel, candidate against a baseline.

  quickcheck_suite.py --stacker X [--decoder Y] --baseline-stacker OLD [--baseline-decoder ND0]
                      [--panels DIR ...] [--par 2] [--jobs 6] [--out DIR]

For each panel the baseline (OLD stacker + ND0 newdecode) is run once and kept in
<panel>/results/base_<stacker md5>_<decoder md5> (stack, v6 cache, audio), then the candidate
(X + Y) is scored against it with the same gates.  --decoder / --baseline-decoder take what
quickcheck.py --decoder takes (t2d, none, v6, a newdecode.py or its directory).  Prints one line
per side (PASS / FAIL and the failed checks) and writes <out>/suite.json, suite.txt and the
contact sheets (<out>/<side>.png).  Exit 1 on any FAIL, 2 when a v6 stage was skipped.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quickcheck import nd_hash, nd_script   # noqa: E402

PY = "/mnt/s/domsday/ldenv/bin/python"
QS = "/mnt/e/domsday_scratch/quickcheck"
PANELS = [f"{QS}/north2", f"{QS}/south_s3000"] + [
    f"{QS}/gen/{s}" for s in ("NationalA", "NationalB", "EcoDisc_S1", "Countryside_S1", "BGB_S1", "UN",
                               "City_Culture1-Side1", "City_Financial1-Side1")]


def md5(path):
    if path is None:
        return "t2d"
    if path.endswith("newdecode.py"):
        return nd_hash(path)[:8]
    if os.path.isdir(path):
        path = os.path.join(path, "stack.py")
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:8]


def run_qc(args, log):
    with open(log, "w") as f:
        return subprocess.run([PY, os.path.join(HERE, "quickcheck.py")] + args, stdout=f, stderr=subprocess.STDOUT).returncode


def one(panel, a, tag):
    side = os.path.basename(panel)
    bnd = nd_script(a.baseline_decoder)
    bdir = os.path.join(panel, "results", f"base_{md5(a.baseline_stacker)}_{md5(bnd)}{a.baseline_suffix}")
    t0 = time.time()
    os.makedirs(os.path.join(panel, "results"), exist_ok=True)
    if not os.path.exists(os.path.join(bdir, "results.json")):
        # a kept stack of the same baseline stacker on this panel (any decoder / panel file): score it into
        # a dir of its own (stack and decode caches linked), never in place of another run's results
        kept = [d for d in sorted(os.listdir(os.path.join(panel, "results")))
                if d.startswith(f"base_{md5(a.baseline_stacker)}") and os.path.isfile(os.path.join(panel, "results", d, "stk.tbc"))
                and not os.path.islink(os.path.join(panel, "results", d, "stk.tbc"))]
        if kept and not os.path.lexists(bdir):
            src = os.path.join(panel, "results", kept[0])
            os.makedirs(bdir)
            for n in os.listdir(src):
                if n.startswith("stk.") or n.startswith("v6cache_") or n == "stack.log":
                    os.symlink(os.path.abspath(os.path.join(src, n)), os.path.join(bdir, n))
            rc = run_qc(["--stacker", a.baseline_stacker, "--panel", panel, "--panel-file", a.panel_file,
                         "--decoder", a.baseline_decoder, "--from-stack", bdir, "--out", bdir], bdir + ".log")
        else:
            rc = run_qc(["--stacker", a.baseline_stacker, "--panel", panel, "--panel-file", a.panel_file,
                         "--decoder", a.baseline_decoder,
                         "--out", bdir, "--keep", "--jobs", str(a.jobs)], bdir + ".log")
        if not os.path.exists(os.path.join(bdir, "results.json")):
            return side, {"verdict": "ERROR", "why": f"baseline run failed (rc {rc}), see {bdir}.log"}
    tb = time.time() - t0
    cdir = os.path.join(panel, "results", f"cand_{tag}{a.baseline_suffix}")
    rc = run_qc(["--stacker", a.stacker, "--panel", panel, "--panel-file", a.panel_file, "--decoder", a.decoder,
                 "--baseline-decoder", a.baseline_decoder, "--baseline", bdir, "--out", cdir, "--keep",
                 "--jobs", str(a.jobs)], cdir + ".log")
    try:
        r = json.load(open(os.path.join(cdir, "results.json")))
    except (OSError, ValueError):
        return side, {"verdict": "ERROR", "why": f"candidate run failed (rc {rc}), see {cdir}.log"}
    fails = [c[0] for c in r["checks"] if c[4] == "FAIL"]
    S = r["summary"]
    key = {k: S.get(k) for k in ("bar_grain_mean", "nd_bar_grain_mean", "y_flat_grain", "nd_flat_grain",
                                 "comb_t2d", "nd_comb", "nd_ghost", "resid_do", "do_err", "flicker_level")}
    for src in ("map_comp", "map_dec", "map_v6"):
        if S.get(src):
            key[src + "_rise"] = S[src].get("rise")
            key[src + "_text"] = S[src].get("text")
    if S.get("err_run"):
        key["err_run_2-3.5"] = S["err_run"][1]
    if r.get("weights"):
        key["w_excess"] = r["weights"].get("max_excess", r["weights"].get("status"))
    if os.path.exists(os.path.join(cdir, "contact.png")):
        shutil.copy(os.path.join(cdir, "contact.png"), os.path.join(a.out, side + ".png"))
    return side, {"verdict": "FAIL" if fails else ("SKIP" if rc == 2 else "PASS"), "fails": fails,
                  "metrics": key, "checks": r["checks"], "results": cdir, "baseline": bdir,
                  "baseline_s": round(tb), "candidate_s": round(r["timing"]["total"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stacker", required=True)
    ap.add_argument("--decoder", default="t2d")
    ap.add_argument("--baseline-stacker", default="/mnt/s/domsday/prefix_backup_2026-10-26/stack_m.py")
    ap.add_argument("--baseline-decoder", default="/mnt/s/domsday/paldec_dev/v6_0")
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--panel-file", default="panel.json", help="e.g. panel_v2.json (raw-capture truth)")
    ap.add_argument("--baseline-suffix", default="", help="use <base dir><suffix> (e.g. _truth2: a baseline "
                    "rescored against panel_v2.json's truth)")
    ap.add_argument("--par", type=int, default=2, help="panels at once (newdecode runs one at a time anyway)")
    ap.add_argument("--jobs", type=int, default=6, help="stacker workers per panel")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    # freeze both decoders once for the whole suite (quickcheck reuses the same frozen copies)
    for k in ("decoder", "baseline_decoder"):
        nd = nd_script(getattr(a, k))
        if nd:
            setattr(a, k, os.path.dirname(nd))
    tag = f"{md5(a.stacker)}_{md5(nd_script(a.decoder))}"
    a.out = a.out or os.path.join(QS, "suite", f"{tag}_{time.strftime('%m%d-%H%M')}")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {}
    with cf.ThreadPoolExecutor(a.par) as ex:
        for side, r in ex.map(lambda p: one(p, a, tag), [p for p in a.panels if os.path.exists(os.path.join(p, a.panel_file))]):
            res[side] = r
            print(f"{side:24} {r['verdict']:5} " + ("; ".join(r.get("fails", [])) or r.get("why", "")), flush=True)
    lines = [f"quickcheck suite: stacker {a.stacker} ({md5(a.stacker)}), decoder {a.decoder}",
             f"  baseline {a.baseline_stacker} ({md5(a.baseline_stacker)}), decoder {a.baseline_decoder}",
             f"  {len(res)} panels in {time.time() - t0:.0f} s", ""]
    for side, r in res.items():
        lines.append(f"{side:24} {r['verdict']:5} " + ("; ".join(r.get("fails", [])) or r.get("why", "")))
    fmt = lambda x: "-" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))
    for side, r in res.items():
        lines += ["", f"== {side} ({r.get('verdict')}; baseline {r.get('baseline_s')} s, candidate {r.get('candidate_s')} s)"]
        for name, c, b, ratio, v in r.get("checks", []):
            lines.append(f"  {name:56} {fmt(c):>10} {fmt(b):>10} {fmt(ratio):>7}  {v}")
    open(os.path.join(a.out, "suite.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(a.out, "suite.json"), "w"), indent=1)
    print("\n".join(lines[:3]))
    print(f"  results: {a.out}")
    if any(r["verdict"] in ("FAIL", "ERROR") for r in res.values()):
        return 1
    return 2 if any(r["verdict"] == "SKIP" for r in res.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
