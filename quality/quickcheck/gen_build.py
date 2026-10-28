# SPDX-License-Identifier: GPL-3.0-or-later
"""Build one generalisation panel from short decodes of every capture of a side.

  gen_build.py <spec.json> <panel dir> [--decode-only] [--par 3]

spec.json (gen_select.py, then edited): {"side", "clv", "segments": [[p0, n, role], ...],
  "runs": [[r0, r1, name, [scored]]], "motion": [[p0, p1]], "captures": [{"name", "ldf" | "full", "spur"}],
  "baseline_stacker", "nd_spec": [...] (optional)}
1. every capture: each segment decoded from its RF (`ld-decode -t 4 --PAL --vits_white_level
   [--V4300D_coherent_subtract] --noEFM --seek p0-3 -l n+6`, `--par` at a time) or, for a capture with
   a full decode ("full": base), cut from it (frames whose picture key is in p0-3 .. p0+n+2)
2. the segments of a capture joined into <panel>/caps/<name>.tbc (+ .tbc.db, .tbc.json, .pcm), fields in
   their pairs, field ids renumbered, picture numbers / time codes as decoded
3. a reference stack of the panel with the baseline stacker (<panel>/refstack/stk; a 1-capture side
   uses the capture itself, "no_stack")
4. panel.json, held-run means from the reference's unscored frames, then prepare_panel.py and
   measure_masters.py as for North
"""
import argparse
import concurrent.futures as cf
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qcmetrics as qm                          # noqa: E402
from extract_panel import PER_FIELD, extract_ref   # noqa: E402
from qcpkg import make_pkg                      # noqa: E402

PY = "/mnt/s/domsday/ldenv/bin/python"
LD = "/mnt/s/domsday/ld-decode"
FB = qm.H * qm.W * 2


def load_meta(base):
    sys.path.insert(0, LD)
    from lddecode.tbcmeta import load_decode_metadata
    return load_decode_metadata(base)


def decode_seg(ldf, spur, p0, n, out):
    if os.path.exists(out + ".log") and ("Took" in open(out + ".log").read() or os.path.exists(out + ".failed")):
        return out
    cmd = [PY, "ld-decode", "-t", "4", "--PAL", "--vits_white_level"] + (["--V4300D_coherent_subtract"] if spur else []) \
        + ["--noEFM", "--seek", str(max(p0 - 3, 1)), "-l", str(n + 6), ldf, out]
    with open(out + ".log", "w") as lf:
        subprocess.run(["nice", "-n", "5"] + cmd, cwd=LD, stdout=lf, stderr=subprocess.STDOUT,
                       env=dict(os.environ, LDDECODE_EFM_GEARSHIFT="0"))
    if "Took" not in open(out + ".log").read():
        open(out + ".failed", "w").close()          # not retried; the segment is left out
    return out


def _pread(fd, n, off):
    # drvfs reads under load fail now and then with ENOMEM (as stack.py's): back off and retry
    import errno
    import time
    for i in range(8):
        try:
            return os.pread(fd, n, off)
        except OSError as e:
            if e.errno != errno.ENOMEM or i == 7:
                raise
            time.sleep(0.5 * 2 ** i)


def read_part(base, frames=None):
    """(meta fields, video bytes, audio bytes, db rows) of a decode, or of frame indices `frames` of it."""
    meta = load_meta(base)
    nfr = len(meta["fields"]) // 2
    frames = list(range(nfr)) if frames is None else frames
    pcm = base + ".pcm"
    spf = os.path.getsize(pcm) // 4 // max(nfr, 1) if os.path.exists(pcm) else 0
    fd = os.open(base + ".tbc", os.O_RDONLY)
    vid = b"".join(_pread(fd, 2 * FB, f * 2 * FB) for f in frames)
    os.close(fd)
    aud = b""
    if spf:
        afd = os.open(pcm, os.O_RDONLY)
        aud = b"".join(_pread(afd, spf * 4, f * spf * 4) for f in frames)
        os.close(afd)
    old = [2 * f + k for f in frames for k in (0, 1)]
    src = sqlite3.connect(f"file:{base}.tbc.db?mode=ro", uri=True)
    schema = [r[0] for r in src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")]
    tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    rows = {t: (None, src.execute(f"SELECT * FROM {t}").fetchall()) for t in ("capture", "pcm_audio_parameters") if t in tables}
    oset = set(old)
    for t in PER_FIELD:
        if t in tables:
            cols = [r[1] for r in src.execute(f"PRAGMA table_info({t})")]
            fi = cols.index("field_id")
            rows[t] = (fi, [r for r in src.execute(f"SELECT * FROM {t}") if r[fi] in oset])
    src.close()
    return meta, old, vid, aud, (schema, rows)


def join_parts(parts, out):
    """Write decode parts back to back as one capture."""
    with open(out + ".tbc", "wb") as f:
        for p in parts:
            f.write(p[2])
    with open(out + ".pcm", "wb") as f:
        for p in parts:
            f.write(p[3])
    fields, base_id = [], 0
    if os.path.exists(out + ".tbc.db"):
        os.unlink(out + ".tbc.db")
    dst = sqlite3.connect(out + ".tbc.db")
    for sql in parts[0][4][0]:
        dst.execute(sql)
    for t, (fi, rs) in parts[0][4][1].items():
        if fi is None and rs:
            dst.executemany(f"INSERT INTO {t} VALUES ({','.join('?' * len(rs[0]))})", rs)
    for meta, old, _, _, (schema, rows) in parts:
        fmap = {o: base_id + i for i, o in enumerate(old)}
        for i, o in enumerate(old):
            fields.append(dict(meta["fields"][o], seqNo=base_id + i + 1))
        for t, (fi, rs) in rows.items():
            if fi is None:
                continue
            rs = [r[:fi] + (fmap[r[fi]],) + r[fi + 1:] for r in rs if r[fi] in fmap]
            if rs:
                dst.executemany(f"INSERT INTO {t} VALUES ({','.join('?' * len(rs[0]))})", rs)
        base_id += len(old)
    dst.execute("UPDATE capture SET number_of_sequential_fields=?", (len(fields),))
    dst.commit()
    dst.close()
    m = {k: v for k, v in parts[0][0].items() if k != "fields"}
    m["fields"] = fields
    m["videoParameters"] = dict(m["videoParameters"], numberOfSequentialFields=len(fields))
    json.dump(m, open(out + ".tbc.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("panel")
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--decode-only", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    pdir = os.path.abspath(a.panel)
    for d in ("seg", "caps"):
        os.makedirs(os.path.join(pdir, d), exist_ok=True)
    clv = bool(spec.get("clv"))
    segs = spec["segments"]
    # 1. decodes
    jobs = []
    with cf.ThreadPoolExecutor(a.par) as ex:
        for c in spec["captures"]:
            if c.get("ldf"):
                for p0, n, _ in segs:
                    out = os.path.join(pdir, "seg", f"{c['name']}_{p0}")
                    jobs.append(ex.submit(decode_seg, c["ldf"], c.get("spur", 0), p0, n, out))
        for j in cf.as_completed(jobs):
            print("decoded", j.result(), flush=True)
    if a.decode_only:
        return
    # 2. join per capture
    for c in spec["captures"]:
        out = os.path.join(pdir, "caps", c["name"])
        if os.path.exists(out + ".tbc.json"):
            continue
        if c.get("ldf"):
            parts = []
            for p0, n, _ in segs:
                b = os.path.join(pdir, "seg", f"{c['name']}_{p0}")
                try:
                    if os.path.getsize(b + ".tbc") >= 8 * FB:
                        parts.append(read_part(b))
                        continue
                except (OSError, ValueError):
                    pass
                print(f"  {c['name']}: segment {p0} did not decode, left out", flush=True)
        else:
            meta = load_meta(c["full"])
            ks = qm.frame_keys(meta["fields"], clv)
            want = [(p0 - 3, p0 + n + 2) for p0, n, _ in segs]
            parts = []
            for lo, hi in want:
                fr = [j for j, k in enumerate(ks) if k is not None and lo <= k <= hi]
                if fr:
                    # a stray misread picture number elsewhere in the file must not stretch the cut
                    m = int(np.median(fr))
                    fr = [j for j in fr if abs(j - m) <= hi - lo + 20]
                    fr = list(range(min(fr), max(fr) + 1))
                    parts.append(read_part(c["full"], fr))
        join_parts(parts, out)
        print("joined", out, flush=True)
    names = [c["name"] for c in spec["captures"]]
    caps = [os.path.join(pdir, "caps", n + ".tbc") for n in names]
    # content-offset check: a capture whose picture numbers are off by one (Countryside ds3) would poison
    # the reference stack, the truth and master_cov; renumber it (CAV) before anything uses it
    if not os.path.exists(os.path.join(pdir, "offsets.json")) and len(names) > 1:
        json.dump({"caps": names, "clv": clv}, open(os.path.join(pdir, "panel.json"), "w"))
        subprocess.run([PY, os.path.join(HERE, "check_offsets.py"), pdir, "--fix"], check=True)
    # 3. reference stack
    no_stack = len(caps) == 1
    ref = os.path.join(pdir, "refstack", "stk")
    if no_stack:
        ref = caps[0][:-4]
    elif not os.path.exists(ref + ".tbc.json"):
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        top, mod, h = make_pkg(spec["baseline_stacker"], os.path.join(pdir, "pkg"))
        cmd = [PY, "-m", mod, "--PAL", "--jobs", os.environ.get("GEN_JOBS", "6"), "--cross_master"] + (["--clv"] if clv else []) + ["-o", ref] + caps
        with open(ref + ".log", "w") as lf:
            subprocess.run(cmd, cwd=top, env=dict(os.environ, PYTHONPATH=top, LDSTACK_GAP_MAX="0", LDSTACK_BLOCK="10"),
                           stdout=lf, stderr=subprocess.STDOUT, check=True)
    # 4. panel.json
    rmeta = load_meta(ref)
    keys = sorted({k for k in qm.frame_keys(rmeta["fields"], clv) if k is not None})
    role_segs = [{"p0": p0, "p1": p0 + n - 1, "role": r} for p0, n, r in segs]
    for s in role_segs:
        if s["role"] == "text":
            s["role"] = "still map text"
    mids = {r: p0 + n // 2 for p0, n, r in segs}
    crops = [[mids[r], r, 300, 200, 300, 140] for r in ("held", "held2", "held3", "text", "saturated", "dark",
                                                       "dropouts", "motion", "bars")
             if r in mids and mids[r] in keys]
    mot = [k for p0, p1 in spec.get("motion", []) for k in range(p0 + 10, p0 + 13) if k in keys]
    # newdecode key pictures (each with 4 neighbours either side, "nd_context"): held, held colour, text,
    # saturated, motion
    mpics = [k for k in (spec.get("map_pics") or []) if k in keys]
    nd = spec.get("nd_spec") or ([m for m in [mids.get("held"), mids.get("held2"), mids.get("held3")] if m in keys]
                                 + mpics[:2] + [m for m in [mids.get("saturated")] if m in keys] + mot[1:2])
    panel = {"side": spec["side"], "caps": names, "clv": clv, "stack_args": ["--clv"] if clv else [],
             "stack_keys": keys, "segments": role_segs, "no_stack": no_stack,
             "runs": [{"r0": r0, "r1": r1, "name": nm, "stacked": [k for k in st if k in keys], "textured": True}
                      for r0, r1, nm, st in spec.get("runs", [])],
             "reference": {"tbc": ref + ".tbc", "clv": clv}, "crop_spec": crops, "nd_spec": nd,
             "motion": mot, "map_ranges": [[s["p0"], s["p1"]] for s in role_segs if "text" in s["role"]],
             "text_crop_pics": mpics[:2] if mpics else ([mids["text"]] if "text" in mids else []),
             "analysis": None, "nd_context": 4}
    if spec.get("map_pics") is not None:
        panel["map_pics"] = mpics
    panel["runs"] = [r for r in panel["runs"] if len(r["stacked"]) >= 2]
    extract_ref(panel, pdir, FB, qm.H, qm.W)
    json.dump(panel, open(os.path.join(pdir, "panel.json"), "w"), indent=1)
    st = spec["baseline_stacker"]
    subprocess.run([PY, os.path.join(HERE, "prepare_panel.py"), pdir, st], check=True)
    if not no_stack:
        subprocess.run([PY, os.path.join(HERE, "measure_masters.py"), pdir, "--frames-per-run", "6"], check=True)
    print("panel", pdir, len(keys), "pictures,", len(names), "captures")


if __name__ == "__main__":
    main()
