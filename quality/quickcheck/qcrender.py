# SPDX-License-Identifier: GPL-3.0-or-later
"""Decode chosen frames of a stacked TBC with ld-chroma-decoder transform2d (in memory).

  qcrender.py <stack.tbc> <out.png> <picture>... [--scale 0.25] [--clv]   contact sheet, for choosing a panel

Library: frames_tbc(tbc, meta, frame_indices, out_base) writes a small TBC + .tbc.json of those
frames; decode(tbc_base) -> list of yuv uint16[3,576,928].
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcmetrics as qm          # noqa: E402

FB = qm.H * qm.W * 2
OW, OH = 928, 576


def load_meta(tbc):
    base = tbc[:-4] if tbc.endswith(".tbc") else tbc
    sys.path.insert(0, "/mnt/s/domsday/ld-decode")
    from lddecode.tbcmeta import load_decode_metadata
    return load_decode_metadata(base)


def frames_tbc(tbc, meta, idx, out_base, frames=None):
    """Write frames `idx` of tbc (or the given uint16 [n,2,H,W] `frames`) as out_base.tbc/.tbc.json."""
    with open(out_base + ".tbc", "wb") as o:
        if frames is not None:
            o.write(np.ascontiguousarray(frames, "<u2").tobytes())
        else:
            fd = os.open(tbc, os.O_RDONLY)
            for i in idx:
                o.write(os.pread(fd, 2 * FB, i * 2 * FB))
            os.close(fd)
    f = meta["fields"]
    m = {k: v for k, v in meta.items() if k != "fields"}
    m["fields"] = [dict(f[2 * i + k], seqNo=2 * j + k + 1) for j, i in enumerate(idx) for k in (0, 1)]
    m["videoParameters"] = dict(meta["videoParameters"], numberOfSequentialFields=len(m["fields"]))
    json.dump(m, open(out_base + ".tbc.json", "w"))


def decode(base, threads=4):
    fb = OW * OH * 3 * 2
    p = subprocess.run(["ld-chroma-decoder", "-q", "-f", "transform2d", "-t", str(threads), "-p", "yuv",
                        "--input-json", base + ".tbc.json", base + ".tbc", "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    n = len(p.stdout) // fb
    return [np.frombuffer(p.stdout[i * fb:(i + 1) * fb], "<u2").reshape(3, OH, OW) for i in range(n)]


def main():
    from PIL import Image, ImageDraw
    from quickcheck import to_rgb8
    tbc, out = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    scale = 0.25
    if "--scale" in args:
        i = args.index("--scale")
        scale = float(args[i + 1])
        del args[i:i + 2]
    pics = [int(x) for x in args if x != "--clv"]
    meta = load_meta(tbc)
    flds = meta["fields"]
    p2f = {}
    for j, k in enumerate(qm.frame_keys(flds, "--clv" in sys.argv)):
        if k is not None and k not in p2f:
            p2f[k] = j
    idx = [p2f[p] for p in pics]
    with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
        frames_tbc(tbc, meta, idx, os.path.join(td, "r"))
        yuv = decode(os.path.join(td, "r"))
    tw, th = int(OW * scale), int(OH * scale)
    cols = min(6, len(pics))
    rows = (len(pics) + cols - 1) // cols
    im = Image.new("RGB", (cols * tw, rows * (th + 12)), (0, 0, 0))
    dr = ImageDraw.Draw(im)
    for n, (p, y) in enumerate(zip(pics, yuv)):
        t = Image.fromarray(to_rgb8(y)).resize((tw, th), Image.BILINEAR)
        x0, y0 = (n % cols) * tw, (n // cols) * (th + 12)
        im.paste(t, (x0, y0 + 12))
        dr.text((x0 + 2, y0), str(p), fill=(255, 255, 0))
    im.save(out)


if __name__ == "__main__":
    main()
