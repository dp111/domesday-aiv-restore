# SPDX-License-Identifier: GPL-3.0-or-later
"""One cheap pass over a stacked side to classify its frames for panel selection.

Reads one 32-line stripe per sampled frame (field 0 lines 100-131, mid-picture, 72 KB of the
1.4 MB frame) in file order: one small forward read per frame instead of 77 GB (seek-bound on a busy HDD).
Per frame and stripe: mean luma, luma std (contrast), chroma amplitude (fsc band, 4fsc sampling),
2-5 MHz luma detail and a 4x8-block thumbnail (for still / identical-frame detection).

  scan_side.py <stack.tbc> <out.npz> [--step N]
"""
import argparse
import os
import sys
import time

import numpy as np

W, H = 1135, 313
STRIPES = ((0, 100),)          # (field in frame, first line)
NL = 32
C0, C1 = 185, 1107               # active video


def stripe_stats(s):
    s = s[:, C0:C1].astype(np.float32)
    # 4fsc sampling: chroma sits at fs/4, so a [1,2,1]/4 cancels it and [1,0,-1]/2 picks it out
    y = (s[:, :-2] + 2 * s[:, 1:-1] + s[:, 2:]) * 0.25
    c = (s[:, :-2] - s[:, 2:]) * 0.5
    hf = y[:, 2:-2] - 0.25 * (y[:, :-4] + 2 * y[:, 2:-2] + y[:, 4:]) * 1.0
    bw = (y.shape[1] // 8) * 8
    th = y[:, :bw].reshape(NL // 4, 4, bw // 8, 8).mean(axis=(1, 3))
    return (float(y.mean()), float(y.std()), float(np.sqrt((c * c).mean())),
            float(np.sqrt((hf * hf).mean())), th.astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tbc")
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()
    fb = W * H * 2
    nfr = os.path.getsize(a.tbc) // (2 * fb)
    fd = os.open(a.tbc, os.O_RDONLY)
    frames = np.arange(0, nfr, a.step)
    st = np.zeros((len(frames), len(STRIPES), 4), np.float32)
    th = None
    t0 = time.time()
    for i, f in enumerate(frames):
        for j, (fld, l0) in enumerate(STRIPES):
            off = (2 * f + fld) * fb + l0 * W * 2
            b = np.frombuffer(os.pread(fd, NL * W * 2, off), "<u2").reshape(NL, W)
            m, sd, ch, hf, t = stripe_stats(b)
            if th is None:
                th = np.zeros((len(frames), len(STRIPES)) + t.shape, np.float16)
            st[i, j] = (m, sd, ch, hf)
            th[i, j] = t
        if i % 2000 == 0:
            print(f"{i}/{len(frames)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    np.savez(a.out, frames=frames, stats=st, thumbs=th)
    print(f"{len(frames)} frames in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
