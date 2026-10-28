# SPDX-License-Identifier: GPL-3.0-or-later
"""Cut the panel's pictures out of every capture into mini captures the stacker reads like the full ones.

  extract_panel.py <panel.json> <index.json> <outdir> [--sequential] [--only NAME,...]

For each capture: the frames of every panel segment plus one trailing guard frame (the stacker's
sequence sieve drops a segment's last frame when the next segment is far away: its median test sees
two neighbours of the next segment) are copied in their original order, fields in their original
pairs, as <outdir>/caps/<capture name>.tbc with
  .tbc.db    the capture's sqlite rows for those fields (every per-field table: field_record, vbi,
             drop_outs, vits_metrics, vitc, closed_caption), field_id renumbered, capture row kept
             with number_of_sequential_fields updated
  .tbc.json  the same fields sliced from the capture's .tbc.json (what the stacker reads when it is
             there), seqNo renumbered
  .pcm       each copied frame's audio block (the stacker reads audio as frame index x samples/frame,
             samples/frame = pcm length // frames, so the blocks are cut the same way)
Picture numbers and VBI stay as decoded, so the stacker indexes and aligns the mini captures as usual.

Disc reads are double-buffered: a reader thread pulls capture k+1's frames (sparse HDD reads, one per
segment) into a 1-deep queue while capture k is written (--sequential: no overlap, for timing).
Also writes <outdir>/ref: the production stack's frames for the panel pictures (ref.npy, uint16
[n,2,H,W], ref_keys.npy) and the held-run temporal means (runmean_<p0>.npy) from frames NOT stacked.
"""
import argparse
import bisect
import json
import os
import queue
import sqlite3
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcmetrics as qm   # noqa: E402

REF_SPAN = 40           # reference frames: up to this far either side of the stacked ones
PER_FIELD = ("field_record", "vbi", "drop_outs", "vits_metrics", "vitc", "closed_caption")


def frame_ranges(cap, segs):
    """Original frame-index ranges (inclusive) to copy for this capture."""
    idx = dict(zip(cap["keys"], cap["fi"]))
    fsorted = sorted(cap["fi"])
    rs = []
    for s in segs:
        fis = [idx[p] for p in range(s["p0"], s["p1"] + 1) if p in idx]
        if fis:
            # guard: up to the next frame with a picture number (frames between without one are
            # copied too, as the capture has them)
            j = bisect.bisect_right(fsorted, max(fis))
            g = fsorted[j] if j < len(fsorted) and fsorted[j] - max(fis) <= 4 else max(fis) + 1
            rs.append([min(fis), min(g, cap["nframes"] - 1)])
    rs.sort()
    out = []
    for a, b in rs:
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def read_capture(cap, ranges, fbytes):
    """The slow part (HDD): every range's fields and audio, plus the metadata."""
    base = cap["base"]
    with open(base + ".tbc.json") as f:
        meta = json.load(f)
    nfr = len(meta["fields"]) // 2
    pcm = base + ".pcm"
    spf = os.path.getsize(pcm) // 4 // nfr if os.path.exists(pcm) else 0
    db = read_db(base + ".tbc.db", ranges)
    vid, aud = [], []
    fd = os.open(base + ".tbc", os.O_RDONLY)
    afd = os.open(pcm, os.O_RDONLY) if spf else None
    try:
        for a, b in ranges:
            n = b - a + 1
            vid.append(os.pread(fd, n * 2 * fbytes, a * 2 * fbytes))
            if afd is not None:
                aud.append(os.pread(afd, n * spf * 4, a * spf * 4))
    finally:
        os.close(fd)
        if afd is not None:
            os.close(afd)
    return meta, db, vid, aud


def read_db(path, ranges):
    """Schema, capture rows and the per-field rows of the copied fields."""
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    schema = [r[0] for r in src.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")]
    tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    rows = {}
    for t in ("capture", "pcm_audio_parameters"):
        if t in tables:
            rows[t] = (None, src.execute(f"SELECT * FROM {t}").fetchall())
    for t in PER_FIELD:
        if t in tables:
            cols = [r[1] for r in src.execute(f"PRAGMA table_info({t})")]
            rows[t] = (cols.index("field_id"), [r for a, b in ranges for r in src.execute(
                f"SELECT * FROM {t} WHERE capture_id=1 AND field_id BETWEEN ? AND ?", (2 * a, 2 * b + 1))])
    src.close()
    return schema, rows


def write_capture(name, cap, ranges, meta, db, vid, aud, outdir):
    fis = [f for a, b in ranges for f in range(a, b + 1)]
    old = [2 * f + k for f in fis for k in (0, 1)]
    ob = os.path.join(outdir, name)
    with open(ob + ".tbc", "wb") as f:
        for v in vid:
            f.write(v)
    if aud:
        with open(ob + ".pcm", "wb") as f:
            for a in aud:
                f.write(a)
    fields = []
    for i, o in enumerate(old):
        fj = dict(meta["fields"][o])
        fj["seqNo"] = i + 1
        fields.append(fj)
    m = dict(meta)
    m["fields"] = fields
    m["videoParameters"] = dict(meta["videoParameters"], numberOfSequentialFields=len(fields))
    with open(ob + ".tbc.json", "w") as f:
        json.dump(m, f)
    # sqlite: same schema, the copied fields' rows, field_id renumbered
    if os.path.exists(ob + ".tbc.db"):
        os.unlink(ob + ".tbc.db")
    schema, rows = db
    dst = sqlite3.connect(ob + ".tbc.db")
    for sql in schema:
        dst.execute(sql)
    fmap = {o: i for i, o in enumerate(old)}
    for t, (fi_col, rs) in rows.items():
        if fi_col is not None:
            rs = [r[:fi_col] + (fmap[r[fi_col]],) + r[fi_col + 1:] for r in rs if r[fi_col] in fmap]
        if rs:
            dst.executemany(f"INSERT INTO {t} VALUES ({','.join('?' * len(rs[0]))})", rs)
    dst.execute("UPDATE capture SET number_of_sequential_fields=?", (len(fields),))
    dst.commit()
    dst.close()
    return len(fis)


def ref_pic2frame(ref):
    """Picture number -> frame index of the reference stack (its frame-stats file, or its VBI)."""
    pic2fr = {}
    if ref.get("frames_tsv"):
        with open(ref["frames_tsv"]) as f:
            next(f)
            for ln in f:
                a = ln.split("\t", 2)
                pic2fr.setdefault(int(a[1]), int(a[0]))
        return pic2fr
    from qcrender import load_meta
    fl = load_meta(ref["tbc"])["fields"]
    for j, k in enumerate(qm.frame_keys(fl, ref.get("clv", False))):
        if k is not None:
            pic2fr.setdefault(k, j)
    return pic2fr


def extract_ref(panel, outdir, fbytes, H, W, runs_only=False):
    """Production stack frames for the panel pictures + held-run means from unstacked frames."""
    ref = panel["reference"]
    pic2fr = ref_pic2frame(ref)
    keys = list(panel["stack_keys"])
    fd = os.open(ref["tbc"], os.O_RDONLY)

    def rd(fr):
        return np.frombuffer(os.pread(fd, 2 * fbytes, fr * 2 * fbytes), "<u2").reshape(2, H, W)

    os.makedirs(os.path.join(outdir, "ref"), exist_ok=True)
    if runs_only:
        arr = np.load(os.path.join(outdir, "ref", "ref.npy"), mmap_mode="r")
    else:
        arr = np.stack([rd(pic2fr[k]) for k in keys])
        np.save(os.path.join(outdir, "ref", "ref.npy"), arr)
        np.save(os.path.join(outdir, "ref", "ref_keys.npy"), np.array(keys))
    # PAL: the subcarrier phase of a line repeats only every 4 frames (fieldPhaseID 1-8), so a
    # still's frames are averaged per phase class (a plain mean would cancel the chroma)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qcrender import load_meta
    pf = load_meta(ref["tbc"])["fields"]
    phase = lambda p: int(pf[2 * pic2fr[p]]["fieldPhaseID"])
    # never average a frame the scorer measures into a reference
    stacked = {k for r in panel.get("runs", []) for k in r["stacked"]}
    for r in panel.get("runs", []):
        st = r["stacked"]
        lo, hi = max(r["r0"], st[0] - REF_SPAN), min(r["r1"], st[-1] + REF_SPAN)
        pics = [p for p in range(lo, hi + 1) if p not in stacked and p in pic2fr]
        fr = [rd(pic2fr[p]) for p in pics]
        # keep only frames identical in content to the stacked ones (8x8-block thumbnails within
        # noise of their median): run ends found from every 3rd frame are a few frames loose
        thumb = lambda f: f[:, qm.RA, qm.CA][:, :280, :920].reshape(2, 35, 8, 115, 8).mean(axis=(2, 4))
        ths = np.array([thumb(f.astype(np.float32)) for f in fr])
        med = np.median([thumb(arr[keys.index(k)].astype(np.float32)) for k in st], axis=0)
        dev = np.sqrt(((ths - med) ** 2).mean(axis=(1, 2, 3)))
        ok = dev < 2.5 * np.median(dev) + 20
        r["ref_dropped"] = [p for p, o in zip(pics, ok) if not o]
        pics = [p for p, o in zip(pics, ok) if o]
        fr = [f for f, o in zip(fr, ok) if o]
        r["phase"] = {str(k): phase(k) for k in st}
        r["ref_frames"], r["floor"], r["prod_noise"] = {}, {}, {}
        for P in sorted(set(r["phase"].values())):
            g = [f for p, f in zip(pics, fr) if phase(p) == P]
            if len(g) < 2:
                continue
            mean = np.mean(g, axis=0)
            np.save(os.path.join(outdir, "ref", f"runmean_{r['r0']}_{P}.npy"), mean.astype(np.float32))
            # the mean's own noise: n frames scatter about it by sigma^2 (n-1)/n, the mean has sigma^2/n
            ms = np.mean([[qm.band_ms(f[i][qm.RA, qm.CA] - mean[i][qm.RA, qm.CA]) for i in (0, 1)] for f in g],
                         axis=(0, 1))
            n = len(g)
            r["ref_frames"][str(P)] = n
            r["floor"][str(P)] = (ms / (n - 1)).tolist()
            r["prod_noise"][str(P)] = np.sqrt(ms * n / (n - 1)).tolist()
    os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("index")
    ap.add_argument("outdir")
    ap.add_argument("--sequential", action="store_true", help="no read/write overlap (timing)")
    ap.add_argument("--only", default="", help="comma list of capture names")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--ref-runs-only", action="store_true", help="only redo the held-run means")
    a = ap.parse_args()
    panel = json.load(open(a.panel))
    index = json.load(open(a.index))
    caps = index["caps"]
    names = [n for n in caps if not a.only or n in a.only.split(",")]
    cdir = os.path.join(a.outdir, "caps")
    os.makedirs(cdir, exist_ok=True)
    H, W = 313, 1135
    fbytes = H * W * 2
    plan = {n: frame_ranges(caps[n], panel["segments"]) for n in names}
    if a.ref_runs_only:
        extract_ref(panel, a.outdir, fbytes, H, W, runs_only=True)
        json.dump(panel, open(os.path.join(a.outdir, "panel.json"), "w"), indent=1)
        return
    t0 = time.time()
    tr, tw = [0.0], [0.0]

    def reader(q):
        for n in names:
            t = time.time()
            item = (n, read_capture(caps[n], plan[n], fbytes))
            tr[0] += time.time() - t
            q.put(item)
        q.put(None)

    if a.sequential:
        for n in names:
            t = time.time()
            meta, db, vid, aud = read_capture(caps[n], plan[n], fbytes)
            tr[0] += time.time() - t
            t = time.time()
            nf = write_capture(n, caps[n], plan[n], meta, db, vid, aud, cdir)
            tw[0] += time.time() - t
            print(f"{n}: {nf} frames in {len(plan[n])} ranges ({time.time() - t0:.0f}s)", flush=True)
    else:
        q = queue.Queue(maxsize=1)      # one capture in flight beside the one being written
        th = threading.Thread(target=reader, args=(q,), daemon=True)
        th.start()
        while True:
            item = q.get()
            if item is None:
                break
            n, (meta, db, vid, aud) = item
            t = time.time()
            nf = write_capture(n, caps[n], plan[n], meta, db, vid, aud, cdir)
            tw[0] += time.time() - t
            del meta, db, vid, aud, item
            print(f"{n}: {nf} frames in {len(plan[n])} ranges ({time.time() - t0:.0f}s)", flush=True)
        th.join()
    print(f"captures: {time.time() - t0:.1f}s wall; reading {tr[0]:.1f}s + writing {tw[0]:.1f}s"
          f" ({'sequential' if a.sequential else 'overlapped'})", flush=True)
    if not a.no_ref and panel.get("reference"):
        t1 = time.time()
        extract_ref(panel, a.outdir, fbytes, H, W)
        json.dump(panel, open(os.path.join(a.outdir, "panel.json"), "w"), indent=1)
        print(f"reference: {time.time() - t1:.1f}s", flush=True)


if __name__ == "__main__":
    main()
