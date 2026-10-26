#!/usr/bin/env python3
"""Per-frame weave / deinterlace of a chroma-decoded MKV, for the PVF.

bwdif decides per pixel whether something moved by looking at the
NEIGHBOURING frames.  On Domesday and the AIV discs those are usually a
different still, so it sees "motion" everywhere and interpolates, keeping
only ~44% of the vertical detail of a frame whose two fields are in fact one
still image (Community South map: jagged diagonals, broken dashed lines).
The AIV live action is film (25p), so its fields are one instant too.

Each frame is therefore scored on its own two fields (combscore: is a line
predicted better by the other field's neighbours, as on a single instant, or
by its own field's, as when the fields differ in time?).  Below THRESH the
frame is woven - the fields exactly as recorded; above, the bwdif result is
used.  Writes a progressive yuv444p16 y4m to stdout (or a FIFO).

  stillweave.py IN.mkv [--start S] [--frames N] [--log FILE] > out.y4m
"""
import argparse, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from combscore import comb_score

THRESH = 0.58     # noise-aware score: static 0.39-0.50 (incl. colour bars); 2 px inter-field motion 0.60-1.24

ap = argparse.ArgumentParser()
ap.add_argument('input'); ap.add_argument('--start', type=float, default=None)
ap.add_argument('--frames', type=int, default=None); ap.add_argument('--log', default=None)
ap.add_argument('--width', type=int, default=928); ap.add_argument('--height', type=int, default=576)
a = ap.parse_args()
W, H = a.width, a.height
FB = W * H * 3 * 2


def reader(vf):
    cmd = ['ffmpeg', '-v', 'error']
    if a.start:
        cmd += ['-ss', str(a.start)]
    cmd += ['-i', a.input]
    if a.frames:
        cmd += ['-frames:v', str(a.frames)]
    # 2026-10-24: masters may be yuv422p10le (co-sited chroma); bring them to 4:4:4 16-bit with a Lanczos
    # chroma upsample (ffmpeg's default bicubic loses ~7 dB of chroma round-trip accuracy).  A yuv444p16
    # master passes through this unchanged.
    up = 'scale=flags=lanczos+accurate_rnd+full_chroma_int:in_chroma_loc=left:in_range=tv:out_range=tv,format=yuv444p16le'
    cmd += ['-vf', (vf + ',' if vf else '') + up]
    cmd += ['-an', '-f', 'rawvideo', '-pix_fmt', 'yuv444p16le', '-']
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FB * 2)


raw = reader(None)
bw = reader('bwdif=mode=0:parity=0')
out = sys.stdout.buffer
out.write(f'YUV4MPEG2 W{W} H{H} F25:1 Ip A259:311 C444p16 XCOLORRANGE=LIMITED\n'.encode())
log = open(a.log, 'w') if a.log else None
n = nd = 0
while True:
    fr = raw.stdout.read(FB); fb = bw.stdout.read(FB)
    if len(fr) < FB or len(fb) < FB:
        break
    Y = np.frombuffer(fr, '<u2', count=W * H).reshape(H, W)
    s = comb_score(Y)
    use = fb if s > THRESH else fr
    nd += s > THRESH
    out.write(b'FRAME\n'); out.write(use)
    if log:
        log.write(f'{n} {s:.3f} {"deint" if s > THRESH else "weave"}\n')
    n += 1
out.flush()
msg = f'stillweave: {n} frames, {nd} deinterlaced ({100 * nd / max(n, 1):.1f}%), {n - nd} woven'
print(msg, file=sys.stderr)
if log:
    log.write('# ' + msg + '\n')
