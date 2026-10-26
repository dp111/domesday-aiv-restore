#!/usr/bin/env python3
"""Per-frame PSNR / SSIM of two raw yuv420p streams, frame n against frame n (fq, 2026-10-26).

  pvf_rawcmp.py <width> <height> <out prefix> <test command> <ref command> [--expect N]
      -> <prefix>.psnr.log, <prefix>.ssim.log in ffmpeg's psnr / ssim stats_file format (n 1-based)

The two commands (run with bash -o pipefail -c) write raw yuv420p frames to stdout; this script runs
them itself, so a command that fails cannot leave the other side blocked on a FIFO.

Replaces ffmpeg's psnr / ssim filters in pvf_quality.sh: their framesync pairs frames by timestamp and
repeats the last frame of a shorter input, so a frame missing on one side silently shifted or padded the
whole comparison (observed across the y4m pipe).  Here frame n is compared with frame n, and the run fails
unless both streams end together, both commands exit 0 (and there are N frames with --expect).  Same
formulas as ffmpeg (8-bit): PSNR per plane and psnr_avg from the plane-size-weighted MSE; SSIM on 8x8
windows stepped by 4 (checked against ffmpeg on the e2e master: PSNR log identical, SSIM within 1e-6).
"""
import math
import os
import signal
import subprocess
import sys

import numpy as np

C1 = int(.01 * .01 * 255 * 255 * 64 + .5)
C2 = int(.03 * .03 * 255 * 255 * 64 * 63 + .5)


def ssim_plane(a, b):
    h, w = a.shape[0] >> 2, a.shape[1] >> 2
    a = a[:h * 4, :w * 4].astype(np.int64); b = b[:h * 4, :w * 4].astype(np.int64)
    blk = lambda x: x.reshape(h, 4, w, 4).sum((1, 3))
    win = lambda s: s[:-1, :-1] + s[1:, :-1] + s[:-1, 1:] + s[1:, 1:]
    s1, s2, ss, s12 = win(blk(a)), win(blk(b)), win(blk(a * a + b * b)), win(blk(a * b))
    var = ss * 64 - s1 * s1 - s2 * s2
    cov = s12 * 64 - s1 * s2
    v = ((2 * s1 * s2 + C1).astype(np.float64) * (2 * cov + C2)
         / ((s1 * s1 + s2 * s2 + C1).astype(np.float64) * (var + C2)))
    return float(v.sum()) / ((h - 1) * (w - 1))


def psnr(mse):
    return math.inf if mse == 0 else 10 * math.log10(255 * 255 / mse)


def compare(fa, fb, w, h, pre):
    """Frames compared; exits on uneven streams."""
    sizes = [(h, w), (h // 2, w // 2), (h // 2, w // 2)]
    fs = sum(y * x for y, x in sizes)
    wt = [y * x / fs for y, x in sizes]
    lp, ls = open(pre + '.psnr.log', 'w'), open(pre + '.ssim.log', 'w')
    n = 0
    sp = ss_ = 0.0
    while True:
        a, b = fa.read(fs), fb.read(fs)
        if len(a) < fs or len(b) < fs:
            if a or b:
                sys.exit(f'FAIL: streams end unevenly after {n} frames ({len(a)} / {len(b)} of {fs} bytes)')
            break
        n += 1
        x, y = np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)
        mse, ssim, o = [], [], 0
        for (ph, pw) in sizes:
            pa, pb = x[o:o + ph * pw].reshape(ph, pw), y[o:o + ph * pw].reshape(ph, pw)
            d = pa.astype(np.int32) - pb
            mse.append(float((d * d).mean()))
            ssim.append(ssim_plane(pa, pb))
            o += ph * pw
        ma = sum(m * k for m, k in zip(mse, wt))
        lp.write(f'n:{n} mse_avg:{ma:0.2f} mse_y:{mse[0]:0.2f} mse_u:{mse[1]:0.2f} mse_v:{mse[2]:0.2f} '
                 f'psnr_avg:{psnr(ma):0.2f} psnr_y:{psnr(mse[0]):0.2f} psnr_u:{psnr(mse[1]):0.2f} '
                 f'psnr_v:{psnr(mse[2]):0.2f} \n')
        sa = sum(s * k for s, k in zip(ssim, wt))
        sdb = math.inf if sa >= 1 else -10 * math.log10(1 - sa)
        ls.write(f'n:{n} Y:{ssim[0]:f} U:{ssim[1]:f} V:{ssim[2]:f} All:{sa:f} ({sdb:f})\n')
        sp += psnr(mse[0]) if mse[0] else 99
        ss_ += ssim[0]
    lp.close(); ls.close()
    if n:
        print(f'pvf_rawcmp: {n} frames paired 1:1, PSNR-Y avg {sp / n:.2f} dB (inf as 99), SSIM-Y avg {ss_ / n:.4f}')
    return n


def main():
    w, h, pre, ct, cr = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    expect = int(sys.argv[sys.argv.index('--expect') + 1]) if '--expect' in sys.argv else None
    fs = w * h * 3 // 2
    procs = [subprocess.Popen(['bash', '-o', 'pipefail', '-c', c], stdout=subprocess.PIPE, bufsize=fs * 4,
                              start_new_session=True) for c in (ct, cr)]
    try:
        n = compare(procs[0].stdout, procs[1].stdout, w, h, pre)
        for name, p in zip(('test', 'reference'), procs):
            if p.wait():
                sys.exit(f'FAIL: {name} command exited with status {p.returncode}')
    finally:
        for p in procs:
            if p.poll() is None:             # the whole command (its own session), not just bash
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    if n == 0:
        sys.exit('FAIL: no frames')
    if expect is not None and n != expect:
        sys.exit(f'FAIL: {n} frames compared, {expect} expected')


if __name__ == '__main__':
    main()
