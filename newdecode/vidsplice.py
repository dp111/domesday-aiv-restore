#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Re-decode the interlaced (moving-video) frames of a newdecode MKV with the
VIDEO model decoder (optcore.solve_fields: per-field unknowns, motion-
compensated temporal priors, noise-weighted data term) and splice them in.

newdecode.py leaves frames whose two fields differ in time (comb_score of the
transform2d init > 0.58) as transform3d.  On the real-noise video bench
(South noise spectrum) the video solver beats transform3d by about
+4.0 dB luma / +4.2 dB chroma, on every motion sequence (2026-10-13).

Stages, each checkpointed in --scratch so a rerun resumes:
  scan    comb_score of every MKV frame -> candidate frames (loose threshold)
  prep    de-smear the candidates (+ margins) into a mini TBC; t2d + t3d decodes
  solve   exact selection (t2d score > 0.58, as newdecode) -> video solve
  splice  MKV frames + solved frames -> new MKV (audio copied)

  vidsplice.py DOC.tbc IN.mkv OUT.mkv --scratch DIR
DOC.tbc = the dropout-corrected TBC newdecode was run on (with .json)."""
import argparse, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palmodel as M
from combscore import comb_score
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

THRESH = 0.58           # newdecode's interlaced test (on the transform2d init)
LOOSE = 0.45            # scan threshold on the MKV output: superset of THRESH
MARGIN = 4              # frames of context each side of a candidate segment
CHUNK = 8               # frames per video solve (+1 context frame each side); 12 -> 8 with W5: GPU memory
VPARAMS = dict(lam_y=0.01, eps_y=2000., lam_cx=0.3, lam_cy=0.1, eps_c=300.,
               lam_t=0.2, eps_t=300., lam_tc=0.3, lam_x=0.0, iters=60, lr=1e4)
LUMA_LPF = 5.2          # as newdecode's default
OW, OH = 928, 576
FB = OW * OH * 3 * 2
FSZ = M.H * M.W * 2

ap = argparse.ArgumentParser()
ap.add_argument('tbc'); ap.add_argument('mkv'); ap.add_argument('out'); ap.add_argument('--scratch', required=True)
ap.add_argument('--expect', type=int, default=None, help='interlaced count newdecode reported (sanity check)')
ap.add_argument('--frames', type=int, default=None, help='side length (default: from the TBC size)')
ap.add_argument('--scan-only', action='store_true')
a = ap.parse_args()
os.makedirs(a.scratch, exist_ok=True)
S = lambda n: os.path.join(a.scratch, n)
log = lambda *x: print('[vidsplice]', *x, file=sys.stderr, flush=True)
nfr = a.frames or os.path.getsize(a.tbc) // (2 * FSZ)
# v6: luma levels from this TBC's declared black/white, as newdecode (v5 used model_params.json's fixed AY/B)
_lv = M.set_levels(json.load(open(a.tbc + '.json'))['videoParameters'])
if _lv:
    log('levels from TBC: black16bIre %.0f white16bIre %.0f -> AY %.6f B %.2f' % (_lv + (M.P['AY'], M.P['B'])))


def _score(b):
    return comb_score(np.frombuffer(b, '<u2').reshape(OH, OW).astype(np.float32))


# ---- scan --------------------------------------------------------------------
if not os.path.exists(S('mkv_scores.npy')):
    t = time.time()
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '16', '-i', a.mkv, '-an', '-vf', 'extractplanes=y',
                          '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'], stdout=subprocess.PIPE, bufsize=OW * OH * 16)
    def frames():
        while True:
            b = p.stdout.read(OW * OH * 2)
            if len(b) < OW * OH * 2:
                return
            yield b
    with ProcessPoolExecutor(16, mp_context=mp.get_context('fork')) as ex:
        sc = np.array(list(ex.map(_score, frames(), chunksize=16, buffersize=256)))
    if p.wait() != 0 or len(sc) != nfr:
        raise SystemExit(f'scan: read {len(sc)} frames of {nfr}')
    np.save(S('mkv_scores.npy'), sc)
    log(f'scan {time.time()-t:.0f} s')
sc = np.load(S('mkv_scores.npy'))
cand = np.flatnonzero(sc > LOOSE)
log(f'{len(cand)} candidates (MKV score > {LOOSE}); {np.sum(sc > THRESH)} above {THRESH}')
if a.scan_only:
    sys.exit(0)

# segments: candidates merged when closer than 2*MARGIN, padded by MARGIN
segs = []
for i in cand:
    if segs and i - segs[-1][1] <= 2 * MARGIN:
        segs[-1][1] = i
    else:
        segs.append([i, i])
segs = [(max(0, s - MARGIN), min(nfr - 1, e + MARGIN)) for s, e in segs]
mini_fr = np.concatenate([np.arange(s, e + 1) for s, e in segs])          # mini frame k -> side frame
pos = {int(f): k for k, f in enumerate(mini_fr)}

# ---- prep --------------------------------------------------------------------
from deghost import channel_inverse, ENABLED as DEGHOST_ON
from deherring import deherring_field
_fd = os.open(a.tbc, os.O_RDONLY)
def _prep(f):
    x = np.frombuffer(os.pread(_fd, 2 * FSZ, int(f) * 2 * FSZ), '<u2').reshape(2, M.H, M.W).astype(np.float64)
    y = np.array([deherring_field(channel_inverse(g))[0] for g in x])
    return np.clip(np.round(y), 0, 65535).astype('<u2').tobytes()

mini = S('mini.tbc')
# v6: the prep stamp records PALDEC_DEGHOST; a resume with the other setting redoes prep (and so the solve)
DG_TAG = 'deghost=%d' % DEGHOST_ON
if os.path.exists(S('prep.done')) and DG_TAG not in open(S('prep.done')).read():
    log('prep.done was made with the other PALDEC_DEGHOST setting: redoing prep and solve')
    for _n in ('prep.done', 'solve.done'):
        if os.path.exists(S(_n)):
            os.remove(S(_n))
if not os.path.exists(S('prep.done')):
    t = time.time()
    with open(mini, 'wb') as w, ProcessPoolExecutor(16, mp_context=mp.get_context('fork')) as ex:
        for b in ex.map(_prep, mini_fr, chunksize=8, buffersize=128):
            w.write(b)
    j = json.load(open(a.tbc + '.json'))
    fl = [j['fields'][2 * int(f) + p] for f in mini_fr for p in (0, 1)]
    for k, d in enumerate(fl):
        d['seqNo'] = k + 1
    j['fields'] = fl; j['videoParameters']['numberOfSequentialFields'] = len(fl)
    json.dump(j, open(mini + '.json', 'w'))
    for dec, extra in (('transform2d', ['--transform-threshold', '0.7']), ('transform3d', [])):
        r = subprocess.run(['ld-chroma-decoder', '-q', '-f', dec] + extra + ['-p', 'y4m', '--input-json', mini + '.json',
                            mini, S(dec + '.y4m')], stderr=subprocess.PIPE)
        if r.returncode:
            raise SystemExit(f'{dec} failed: {r.stderr[-500:]}')
    open(S('prep.done'), 'w').write(f'{len(mini_fr)} frames in {len(segs)} segments {DG_TAG}\n')
    log(f'prep {len(mini_fr)} frames in {len(segs)} segments, {time.time()-t:.0f} s')
os.close(_fd)


class Y4M:
    def __init__(self, p):
        self.fd = os.open(p, os.O_RDONLY)
        h = os.pread(self.fd, 4096, 0); self.hlen = h.index(b'\n') + 1; self.header = h[:self.hlen]
        n = (os.fstat(self.fd).st_size - self.hlen) // (6 + FB)
        assert n == len(mini_fr), (p, n, len(mini_fr))
    def frame(self, k):
        b = os.pread(self.fd, FB, self.hlen + k * (6 + FB) + 6)
        return np.frombuffer(b, '<u2').reshape(3, OH, OW).astype(np.float32)

Y2, Y3 = Y4M(S('transform2d.y4m')), Y4M(S('transform3d.y4m'))

# ---- solve -------------------------------------------------------------------
if not os.path.exists(S('solve.done')):
    # score before torch is imported: the pool forks
    with ProcessPoolExecutor(16, mp_context=mp.get_context('fork')) as ex:
        s2 = np.array(list(ex.map(_score, (Y2.frame(pos[int(f)])[0].astype('<u2').tobytes() for f in cand), chunksize=8)))
    inter = cand[s2 > THRESH]
    log(f'{len(inter)} interlaced frames (t2d score > {THRESH})' + (f'; newdecode reported {a.expect}' if a.expect else ''))
    np.save(S('inter.npy'), inter)
    from noisepsd import estimate as noise_estimate
    # v6: noise spectrum as newdecode (PALDEC_NOISE=active: active picture after the deghost inverse, guarded by
    # the blank-line estimate; vbi = v5); measured before torch (the estimator forks)
    _psd, _nmsg = noise_estimate(a.tbc, os.environ.get('PALDEC_NOISE', 'active'))
    log(_nmsg)
    import torch, optcore as O
    # 2026-10-14 combined candidate: weights / priors / Hanover / luma roll-off from
    # optcore.decoder_config (env PALDEC_NW, PALDEC_PRIORS, PALDEC_HANOVER, PALDEC_LUMA_LPF)
    CFG = O.decoder_config(_psd)
    NOISE_W = CFG['noise_w']
    VP = dict(VPARAMS); VP.update(CFG['video']); LUMA_LPF = CFG['luma_lpf']
    log(f"config: k {CFG['k']:.3f} video {CFG['video']} hanover {CFG['hanover_video']:.2f} luma_lpf {LUMA_LPF}")
    _f = np.fft.rfftfreq(OW, 1 / 17.734475)
    _lp = (0.5 + 0.5 * np.cos(np.pi * np.clip((_f - (LUMA_LPF - 0.4)) / 0.8, 0, 1))).astype(np.float32)
    mfd = os.open(mini, os.O_RDONLY)
    def fields(ks):            # mini-TBC frames -> (2*len, H, W) float32 fields
        return np.concatenate([np.frombuffer(os.pread(mfd, 2 * FSZ, k * 2 * FSZ), '<u2').reshape(2, M.H, M.W)
                               for k in ks]).astype(np.float32)
    runs, cur = [], []
    for f in inter:
        if cur and f == cur[-1] + 1 and pos[int(f)] == pos[int(cur[-1])] + 1:
            cur.append(int(f))
        else:
            if cur: runs.append(cur)
            cur = [int(f)]
    if cur: runs.append(cur)
    t = time.time(); done = 0
    with open(S('repl.yuv'), 'wb') as w:
        for r in runs:
            for k in range(0, len(r), CHUNK):
                core = r[k:k + CHUNK]
                ks = list(range(pos[core[0]] - 1, pos[core[-1]] + 2))   # one context frame each side (margins exist)
                tb = fields(ks)
                C, cu, cv = O.field_setup(tb, range(len(tb)))
                I = np.array([Y3.frame(kk)[:, p::2] for kk in ks for p in (0, 1)])
                fl = O.estimate_flow_gpu(I[:, 0], lag=2)
                o = O.solve_fields(C, cu, cv, I, [p for _ in ks for p in (0, 1)], flow=fl, noise_w=NOISE_W,
                                   hanover=CFG['hanover_video'], **VP)
                for c in range(1, len(ks) - 1):
                    x = np.empty((3, OH, OW), np.float32)
                    x[:, 0::2] = o[2 * c]; x[:, 1::2] = o[2 * c + 1]
                    if LUMA_LPF:
                        x[0] = np.fft.irfft(np.fft.rfft(x[0], axis=1) * _lp, OW, axis=1)
                    w.write(np.clip(np.round(x), 0, 65535).astype('<u2').tobytes())
                done += len(core)
            log(f'solved {done}/{len(inter)} ({time.time()-t:.0f} s)')
    os.close(mfd)
    open(S('solve.done'), 'w').write(f'{len(inter)} frames\n')
inter = np.load(S('inter.npy'))
assert os.path.getsize(S('repl.yuv')) == len(inter) * FB

# ---- splice ------------------------------------------------------------------
t = time.time()
rd = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '16', '-i', a.mkv, '-an', '-f', 'rawvideo',
                       '-pix_fmt', 'yuv444p16le', '-'], stdout=subprocess.PIPE, bufsize=FB * 4)
wr = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', '-', '-i', a.mkv,
                       '-map', '0:v', '-map', '1:a', '-c:v', 'ffv1', '-level', '3', '-slices', '16', '-slicecrc', '1',
                       '-c:a', 'copy', a.out], stdin=subprocess.PIPE)
wr.stdin.write(Y2.header)
rfd = os.open(S('repl.yuv'), os.O_RDONLY)
ri = {int(f): k for k, f in enumerate(inter)}
n = 0
while True:
    b = rd.stdout.read(FB)
    if len(b) < FB:
        break
    if n in ri:
        b = os.pread(rfd, FB, ri[n] * FB)
    wr.stdin.write(b'FRAME\n'); wr.stdin.write(b)
    n += 1
    if n % 5000 == 0:
        log(f'splice {n}/{nfr} ({time.time()-t:.0f} s)')
wr.stdin.close()
if rd.wait() or wr.wait() or n != nfr:
    raise SystemExit(f'splice failed: {n} frames, ffmpeg rc {rd.returncode}/{wr.returncode}')
log(f'done: {len(inter)} frames replaced, {n} written, {time.time()-t:.0f} s')
