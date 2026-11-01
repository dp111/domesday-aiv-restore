#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stacked TBC -> chroma-decoded yuv444p16 y4m with the model-based decoder.

  1. de-smear the TBC (deghost.py: the channel's ~0.4 us edge smear) into a
     scratch TBC
  2. classify frames: a frame whose +-2 neighbours are the same scene is
     video (transform3d's 8-field window is valid); others are stills
  3. starting decodes from ld-chroma-decoder: transform2d (threshold 0.7)
     for stills, transform3d for video
  4. stills are refined by the frame-mode model decoder on the GPU
     (optcore.solve, tuned PARAMS); video keeps transform3d for now
  5. y4m to OUT (file or '-')

  newdecode.py STACK.tbc OUT.y4m --scratch DIR [--start F] [--frames N]
"""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optcore as O, palmodel as M
from deghost import channel_inverse, ENABLED as DEGHOST_ON
from deherring import deherring_field, STAMP as DEHERRING_STAMP
from concurrent.futures import ProcessPoolExecutor
from noisepsd import noise_psd, noise_psd_active, combine as noise_combine, estimate as noise_estimate

# tuned on a bench with noise shaped to a real stack's spectrum,
# with the data term weighted by 1/noise power per frequency (optcore.noise_weights)
# 2026-10-14 combined candidate: the priors, noise weights and Hanover stage come from
# optcore.decoder_config (env switches PALDEC_NW / PALDEC_PRIORS / PALDEC_HANOVER /
# PALDEC_LUMA_LPF, see there), set below once the stack's noise spectrum is known.
PARAMS = dict(lam_hf=0, iters=60, lr=1e4, method='lbfgs')
CORR_MIN = 0.5
BATCH = 8
OW, OH = O.OW, O.OH
FB = OW * OH * 3 * 2

ap = argparse.ArgumentParser()
ap.add_argument('tbc'); ap.add_argument('out'); ap.add_argument('--scratch', required=True)
ap.add_argument('--luma_lpf', type=float, default=None, help='MHz centre of a cosine roll-off (pass below -0.4, stop above +0.4); 0 = off.  Default 5.2: the discs carry no picture above ~5 MHz, and the 5-6 MHz band holds disc noise plus the channel ringing that follows sharp edges (Test Card F: -20%% ringing, deeper lines)')
ap.add_argument('--video', type=int, default=int(os.environ.get('PALDEC_VIDEO', '0')), help='1: solve interlaced frames inline with optcore.solve_fields (as vidsplice.py does afterwards); 0: leave them transform3d')
ap.add_argument('--dropout_mask', type=int, default=int(os.environ.get('PALDEC_DOMASK', '0')), help='1: the solvers give the samples listed as dropOuts in STACK.tbc.json (dilated by DOPAD) zero data weight, so the priors fill them from nearby pixels and a held run from its other frames (v5, for single-copy sides; the input is still the dropout-corrected TBC, whose line copies are only the starting point)')
ap.add_argument('--start', type=int, default=0); ap.add_argument('--frames', type=int, default=None)
a = ap.parse_args()
base = a.tbc[:-4]
# Luma levels from this TBC's declared black/white (2026-10-24): composite = AY*Y + B maps Y 4096 (8-bit 16)
# to black16bIre and Y 60160 (235) to white16bIre.  Real ld-decode TBCs declare the black they measured
# (~16060 on South, not 16384), so the fixed AY/B of model_params.json put the porch 1.9 levels below 16;
# for the fq benchmark's TBCs (16384/54016) this gives exactly the adopted 0.671233/13634.63.
# PALDEC_LEVELS=params keeps the json's AY/B.  (v6: palmodel.set_levels, shared with vidsplice)
_lv = M.set_levels(json.load(open(base + '.tbc.json'))['videoParameters'])
if _lv:
    print('levels from TBC: black16bIre %.0f white16bIre %.0f -> AY %.6f B %.2f' % (_lv + (M.P['AY'], M.P['B'])), file=sys.stderr, flush=True)
os.makedirs(a.scratch, exist_ok=True)
fd = os.open(a.tbc, os.O_RDONLY)
FSZ = M.H * M.W * 2
nfr_all = os.fstat(fd).st_size // (2 * FSZ)
end = nfr_all if a.frames is None else min(nfr_all, a.start + a.frames)
nfr = end - a.start
log = lambda *x: print('[newdecode]', *x, file=sys.stderr, flush=True)

# noise spectrum of THIS stack -> data-term weights and prior scale k
# ZIP_NOISE_FROM: optional other TBC for the noise spectrum (clip tests: use the full side's stack)
# v6 (PALDEC_NOISE=active, default): measured in the ACTIVE PICTURE after the deghost inverse (noisepsd.
# noise_psd_active: same-phase field differences in static blocks), guarded by the blank-line estimate (also
# after the inverse; noisepsd.combine).  The blank lines carry only the primary master (the cross-master blend works on
# the picture), so on 3-master sides v5's estimate (PALDEC_NOISE=vbi) overstated the picture's noise and
# with it every prior: fq benchmark Test Card stack: true noise sigma 385-398 (flat areas), active 366, blank
# lines 1044 (raw 888).  The guard covers clips without static content (CAV stills only).
_nsrc = os.environ.get('ZIP_NOISE_FROM', a.tbc)
NOISE_MODE = os.environ.get('PALDEC_NOISE', 'active')
# v6 agent 10-27 (noisepsd.estimate: spatial floor, flattened-block exclusion, temporal/spatial agreement, fallbacks);
# PALDEC_NOISE_EST=v60 keeps v6_0's path (with PALDEC_NOISE_SFLOOR=0 PALDEC_NOISE_FLAT=0 PALDEC_NOISE_QCAP=2: exact v6_0)
NOISE_EST = os.environ.get('PALDEC_NOISE_EST', 'v6')
if NOISE_EST == 'v6':
    _psd, _nmsg = noise_estimate(_nsrc, NOISE_MODE)
    print('[newdecode] ' + _nmsg, file=sys.stderr, flush=True)
elif NOISE_MODE == 'vbi':
    _psd = noise_psd(_nsrc, nfields=200)
else:
    _ni = {}
    _pa = noise_psd_active(_nsrc, info=_ni); _pv = noise_psd(_nsrc, nfields=200, inverse=True)
    _psd, _qh = noise_combine(_pa, _pv)
    print('[newdecode] noise: active sigma %.0f, blank lines %.0f, used %.0f (5.5-8 MHz power ratio %.3f)  %s'
          % (O.noise_sigma(_pa), O.noise_sigma(_pv), O.noise_sigma(_psd), _qh, _ni), file=sys.stderr, flush=True)
np.save(os.path.join(a.scratch, 'noise_psd.npy'), _psd)
CFG = O.decoder_config(_psd)
# 2026-10-28 (herring integration): the v6 estimate raises sigma where v6_0's cap held it low (City_Culture1 559 vs
# 327), and the video solve's TEMPORAL priors scale with it: more neighbour mixing, motion ghosting 0.42 -> 0.51.
# The video temporal priors are capped at the capped estimate's strength (PALDEC_VID_TCAP=0: off).
import noisepsd as _np6
if NOISE_EST == 'v6' and os.environ.get('PALDEC_VID_TCAP', '1') == '1' and _np6.LAST_CAPPED is not None:
    _cv = O.decoder_config(_np6.LAST_CAPPED)['video']
    _mx = float(os.environ.get('PALDEC_VID_TCAP_MIX', 1.0))   # 1: the capped strength; 0.5: geometric mean of the two
    for _q in ('lam_t', 'lam_tc', 'lam_t4'):
        if _q in CFG['video'] and _q in _cv and _cv[_q] < CFG['video'][_q]:
            CFG['video'][_q] = CFG['video'][_q] ** (1 - _mx) * _cv[_q] ** _mx
    if os.environ.get('PALDEC_FILM_TCAP', '1') == '1':
        # the same for solve_film's temporal priors (BGB_S1 moving film: ghost 0.29 vs v6_0 0.22)
        _cf = O.decoder_config(_np6.LAST_CAPPED)['film']
        for _q in ('lam_t', 'lam_tc'):
            if _q in CFG['film'] and _q in _cf:
                CFG['film'][_q] = min(CFG['film'][_q], _cf[_q])
    print('[newdecode] video temporal priors capped at sigma %.0f: %s' % (O.noise_sigma(_np6.LAST_CAPPED),
          {q: round(CFG['video'][q], 4) for q in ('lam_t', 'lam_tc', 'lam_t4') if q in CFG['video']}), file=sys.stderr, flush=True)
NOISE_W = CFG['noise_w']
NOISE_WV = NOISE_W                      # the inline video solve's weights
# 2026-10-27 (herring integration) region-adaptive data weight (PALDEC_NW_ADAPT=1; default OFF, see below): the in-band reference
# trades detail against grain - p75 keeps map lines and text sharp (North panel rise x1.00 vs v6_0, min x1.03) but
# grains flat saturated colour (South bars x1.7); min the reverse.  So the stills / film / held solves get the p75
# weights, scaled down per pixel to the min level (data weight / sqrt(R), R = p75 / min in band) where the init's
# luma is flat away from any edge; moving film (solve_film) and video keep the min weights.  PALDEC_NW_ADAPT=0: the global PALDEC_NW_REF.
# 10-28: default off -- the p75 detail weight raised map error vs held-run truth (north2 held text pages: text
# area +7.5%, paper +2.4%; the user saw more noise around map text); min weights everywhere as v6_0
NW_ADAPT = os.environ.get('PALDEC_NW_ADAPT', '0') == '1' and O.NW_MODE == 'rec' and O.NW_NORM == 'inband'
if NW_ADAPT:
    _ref0 = O.NW_REF
    O.NW_REF = 'min'; _wmin = O.noise_weights(_psd, floor=0.00625, ceil=0.025, norm=O.NW_NORM)
    O.NW_REF = os.environ.get('PALDEC_NW_ADAPT_REF', 'p75'); _wdet = O.noise_weights(_psd, floor=0.00625, ceil=0.025, norm=O.NW_NORM)
    O.NW_REF = _ref0
    _fb = np.fft.rfftfreq(OW, 1 / 17.734475)
    NW_R = float(np.median((_wdet / _wmin)[(_fb > 0.2) & (_fb < 5.0)]))
    NOISE_W, NOISE_WV = _wdet, _wmin
    print('[newdecode] region-adaptive data weight: detail %s, flat min (R %.2f)' % (os.environ.get('PALDEC_NW_ADAPT_REF', 'p75'), NW_R), file=sys.stderr, flush=True)
# flat / detail thresholds on the init's smoothed luma gradient, relative to the stack's noise sigma
NW_G0 = float(os.environ.get('PALDEC_NW_G0', 1.0)) * O.noise_sigma(_psd); NW_G1 = float(os.environ.get('PALDEC_NW_G1', 3.0)) * O.noise_sigma(_psd)
REGION_W = {}
NW_DIL = int(os.environ.get('PALDEC_NW_DIL', 15))     # detail-zone dilation (px, odd)
NW_FG0 = float(os.environ.get('PALDEC_NW_FILM_G0', 4.0)) * O.noise_sigma(_psd); NW_FG1 = float(os.environ.get('PALDEC_NW_FILM_G1', 8.0)) * O.noise_sigma(_psd)
REGION_WF = {}
def region_weight(Y, g0=None, g1=None):
    """(OH, OW) per-pixel data weight: 1 near detail, 1/sqrt(NW_R) in flat areas (gradient of the init's luma
    pre-smoothed (1, 2) px, then sigma 2: below NW_G0 flat, above NW_G1 detail, both x the noise sigma; bars / map
    paper 66-80 % flat, photos 10 %), the detail zone dilated by 7 px so thin lines / text keep it"""
    from scipy.ndimage import gaussian_filter, maximum_filter
    g = gaussian_filter(np.hypot(*np.gradient(gaussian_filter(Y.astype(np.float32), (1, 2)))), 2.0)
    g0 = NW_G0 if g0 is None else g0; g1 = NW_G1 if g1 is None else g1
    d = maximum_filter(np.clip((g - g0) / (g1 - g0), 0, 1), size=NW_DIL)
    return (1 / np.sqrt(NW_R) + (1 - 1 / np.sqrt(NW_R)) * d).astype(np.float32)
PARAMS.update(CFG['still']); PARAMS['hanover'] = CFG['hanover']
FILM = dict(CFG['film'])
_vts = float(os.environ.get('PALDEC_VID_TSCALE', 1.0))     # video temporal priors x this
for _q in ('lam_t', 'lam_tc', 'lam_t4'):
    if _q in CFG['video']: CFG['video'][_q] *= _vts
if a.luma_lpf is None:
    a.luma_lpf = CFG['luma_lpf']
log('held stills (PALDEC_HELD): %s  min run %d  T %.1f  block %d  iters %d' % (O.HELD, O.HELD_MIN, O.HELD_T, O.HELD_BLOCK, O.HELD_ITERS))
log('config: k %.3f (noise %.0f)  stills %s  film %s  hanover %.2f  luma_lpf %s' % (CFG['k'], CFG['k'] * 928, CFG['still'], FILM, CFG['hanover'], a.luma_lpf))
log('noise weights: 5.5-7 MHz mean %.2f (1 = average band)' % NOISE_W[int(5.5/17.734475*928):int(7/17.734475*928)].mean())

# ---- 1+2: de-smear into scratch TBC, coarse luma for classification -----------
T0 = time.time()
# The de-smeared TBC is cached beside the scratch dir (a full side takes ~35 min to
# build), so a retry after a failure reuses it; the stamp ties it to its source.
dg = os.path.join(os.path.dirname(os.path.abspath(a.scratch)), 'dg_cache.tbc')
_st = os.stat(a.tbc)
STAMP = dict(src=os.path.abspath(a.tbc), size=_st.st_size, mtime=_st.st_mtime, start=a.start, end=end, deghost=DEGHOST_ON,
             deherring=DEHERRING_STAMP)
def _stamp_ok():
    try:
        return json.load(open(dg + '.stamp')) == STAMP and os.path.getsize(dg) == nfr * 2 * FSZ
    except (OSError, ValueError):
        return False
coarse = []
def _prep(buf_n):
    """de-smear then remove the disc's FM-beat herringbone, one chunk of fields"""
    buf, n = buf_n
    x = np.frombuffer(buf, '<u2').reshape(n * 2, M.H, M.W).astype(np.float64)
    y = np.array([deherring_field(channel_inverse(f))[0] for f in x])
    c = [y[2 * i, 30:298, 192:1096].reshape(67, 4, 113, 8).mean((1, 3)) for i in range(n)]
    return np.clip(np.round(y), 0, 65535).astype('<u2').tobytes(), c

import multiprocessing as _mp
# fork explicitly: Python 3.14's default (forkserver) re-runs this script in each
# worker; fork is safe here because CUDA is not initialised until after this pass
def _coarse(buf_n):
    buf, n = buf_n
    y = np.frombuffer(buf, '<u2').reshape(n * 2, M.H, M.W)
    return [y[2 * i, 30:298, 192:1096].astype(np.float64).reshape(67, 4, 113, 8).mean((1, 3)) for i in range(n)]
CH = 32
def chunks(rfd, first, last):
    for f0 in range(first, last, CH):
        n = min(CH, last - f0)
        yield os.pread(rfd, n * 2 * FSZ, f0 * 2 * FSZ), n
if _stamp_ok() and os.path.exists(dg + '.coarse.npy'):
    coarse = list(np.load(dg + '.coarse.npy'))
    log('reusing de-smeared TBC ' + dg)
elif _stamp_ok():
    log('reusing de-smeared TBC ' + dg + ' (re-reading it for the classifier)')
    dfd = os.open(dg, os.O_RDONLY)
    with ProcessPoolExecutor(max_workers=8, mp_context=_mp.get_context('fork')) as ex:
        for c in ex.map(_coarse, chunks(dfd, 0, nfr), buffersize=16):
            coarse.extend(c)
    os.close(dfd)
else:
    if os.path.exists(dg + '.stamp'):
        os.remove(dg + '.stamp')
    with open(dg, 'wb') as w, ProcessPoolExecutor(max_workers=8, mp_context=_mp.get_context('fork')) as ex:
        # buffersize bounds the read-ahead: without it map() submits every chunk at once,
        # i.e. reads the whole TBC into memory (ENOMEM on a full side, 2026-10-13)
        for out_b, c in ex.map(_prep, chunks(fd, a.start, end), buffersize=16):
            w.write(out_b); coarse.extend(c)
    json.dump(STAMP, open(dg + '.stamp', 'w'))
np.save(dg + '.coarse.npy', np.array(coarse))
os.close(fd)
j = json.load(open(base + '.tbc.json'))
# v5 dropout masks: field index (relative to --start) -> list of (field row, x0, x1, fx0, fx1) in solver
# coordinates: x0..x1 the padded solver mask (clipped to the picture), fx0..fx1 the flagged span (v6: unclipped,
# for doi_mask; v5 shrank the clipped padded span back, losing up to DOPAD-DOI_PAD flagged columns at the edges)
DOPAD = int(os.environ.get('PALDEC_DOPAD', '6'))
DO_VMIN = int(os.environ.get('PALDEC_DO_VMIN', '3'))   # shortest interlaced run whose solve gets the mask   # deghost/deherring spread a dropout edge a few samples
DO = {}
if a.dropout_mask:
    for k, fj in enumerate(j['fields'][2 * a.start:2 * end]):
        d = fj.get('dropOuts')
        if not d:
            continue
        for x0, x1, L in zip(d['startx'], d['endx'], d['fieldLine']):
            r = L - 1 - O.LINE0                     # fieldLine is 1-based (checked on Financial1)
            fx0, fx1 = x0 - O.X0, x1 - O.X0
            x0, x1 = max(0, fx0 - DOPAD), min(OW, fx1 + DOPAD)
            if 0 <= r < OH // 2 and x1 > x0:
                DO.setdefault(k, []).append((r, x0, x1, fx0, fx1))
    log('dropout mask: %d fields with dropouts in the picture area, %d segments (pad %d)' % (len(DO), sum(map(len, DO.values())), DOPAD))
def fmask(k):
    m = np.ones((OH // 2, OW), np.float32)
    for r, x0, x1, _, _ in DO.get(k, ()):
        m[r, x0:x1] = 0
    return m
def dmask(idx):
    """frame masks (len(idx), OH, OW) for the solvers, or None when masking is off"""
    if not a.dropout_mask:
        return None
    m = np.empty((len(idx), OH, OW), np.float32)
    for n, i in enumerate(idx):
        m[n, 0::2] = fmask(2 * i); m[n, 1::2] = fmask(2 * i + 1)
    return m
def dmask_solve(idx, film=False):
    """dmask times the region-adaptive data weight (NW_ADAPT), for the stills / held solves; film=True: the film
    map (detail = strong line-art edges only, NW_FILM_G0/G1)"""
    m = dmask(idx)
    if not NW_ADAPT:
        return m
    r = np.stack([(REGION_WF if film else REGION_W)[i] for i in idx])
    return r if m is None else m * r
# v5 dropout interpolation (2026-10-24, user: "use the line before and the line afterwards"): outside held
# stills, the masked solve's dropout pixels are replaced by (PALDEC_DOI=replace) or averaged with
# (PALDEC_DOI=mix) a vertical interpolation of the decoded picture between the nearest clean rows above and
# below (frame rows +-1..4 for stills / film, same-field rows +-2..8 for interlaced video).  Held stills keep
# the solve (it fills from the run's other frames, which beats any spatial fill).  fq dofill test: see
# quality/dofill (dropout luma err 8.96 -> 8.03, never worse per segment); default replace, off = plain masked solve.
DOI = os.environ.get('PALDEC_DOI', 'replace')
DOI_PAD = int(os.environ.get('PALDEC_DOI_PAD', '2'))
DOI_FB_LUMA = os.environ.get('PALDEC_DOI_FB', 'solve') == 'solve'   # v6; 'interp' = v5 (interpolate it too)
# Temporal part (2026-10-25, user: "if that part of the frame doesn't move we can use the previous and next
# frame"): after the vertical interpolation, a dropout pixel takes the mean of the previous / next frame's
# pixel wherever that neighbour is clean there and locally static - box (13x33) mean |dY| and |dC| over the
# pixels clean in both frames < PALDEC_DOT levels (8-bit; 0 = off).  fq dofill: 1.5 never worse per segment.
# v6: the threshold is noise-relative: PALDEC_DOT x sigma / DOT_REF, sigma = the stack's picture noise (16-bit, the
# spectrum above; DOT_REF 600 = the fq benchmark's typical picture noise, where 1.5 was tuned)
DOT = float(os.environ.get('PALDEC_DOT', '1.5'))
DOT_REF = float(os.environ.get('PALDEC_DOT_REF', '600'))
DOT_T = DOT * O.noise_sigma(_psd) / DOT_REF if NOISE_MODE != 'vbi' and os.environ.get('PALDEC_DOT_ABS', '0') != '1' else DOT
from scipy.ndimage import uniform_filter as _uf
def _boxmean(v, w):
    return _uf(v * w, (13, 33), mode='constant') / np.maximum(_uf(w, (13, 33), mode='constant'), 1e-6)
def doi_mask(i):
    m = np.zeros((OH, OW), bool)
    for p in (0, 1):
        for r, _, _, fx0, fx1 in DO.get(2 * i + p, ()):
            # the flagged span + DOI_PAD (v6: from the unclipped span)
            m[2 * r + p, max(0, fx0 - DOI_PAD):min(OW, fx1 + DOI_PAD)] = True
    return m
def doi_fill(x, m, offs):
    """x (3, OH, OW) float32; returns x with m's pixels linearly interpolated between the nearest unmasked rows"""
    va = np.zeros_like(x); vb = np.zeros_like(x); da = np.full(m.shape, np.inf); db = np.full(m.shape, np.inf)
    for d in sorted(offs, key=abs):
        r0, r1 = max(0, -d), min(OH, OH - d)
        ok = np.zeros(m.shape, bool); ok[r0:r1] = ~m[r0 + d:r1 + d]
        sh = np.zeros_like(x); sh[:, r0:r1] = x[:, r0 + d:r1 + d]
        if d < 0:
            b = ok & m & np.isinf(da); va[:, b] = sh[:, b]; da[b] = -d
        else:
            b = ok & m & np.isinf(db); vb[:, b] = sh[:, b]; db[b] = d
    fa, fb = np.isfinite(da), np.isfinite(db)
    w = np.where(fa & fb, db / np.where(fa & fb, da + db, 1), fa.astype(np.float64)).astype(np.float32)
    f = va * w + vb * (1 - w); sel = m & (fa | fb)
    y = x.copy(); y[:, sel] = f[:, sel]
    return y
j['fields'] = j['fields'][2 * a.start:2 * end]
j['videoParameters']['numberOfSequentialFields'] = 2 * nfr
json.dump(j, open(dg + '.json', 'w'))
c = np.array(coarse); c = c - c.mean((1, 2), keepdims=True); sc = np.sqrt((c ** 2).mean((1, 2))) + 1
corr = (c[1:] * c[:-1]).mean((1, 2)) / (sc[1:] * sc[:-1])
same = np.concatenate([[False], corr > CORR_MIN, [False]])
# v4: the coarse test above links DIFFERENT one-frame photos of similar
# layout (sky over ground: coarse corr 0.53-0.71) into film runs, which solve_film then mixes
# (cross-colour checker, ghosts, lost colour; ~27% of South's coarse links).  A link must also pass
# a shift-tolerant fine-detail correlation (field-0 luma in 2x4-sample blocks, high-passed at sigma
# 4 blocks, Hann-windowed, max normalised xcorr over +-12 field lines / +-48 samples) >= FINE_MIN.
# Different photos 0.02-0.39, same picture 0.59-0.999.  Frames whose fine-feature rms is below
# FINE_EMIN (black/flat: 30-70; bars 260, photos/maps 680-2400) give no meaningful correlation:
# their coarse link is kept.  ZIP_FINE=0 disables (= v3 behaviour).
FINE_MIN = float(os.environ.get('ZIP_FINE', 0.4))
FINE_EMIN = float(os.environ.get('ZIP_FINE_EMIN', 150.))
FINE = np.full(len(same), -1.0)
def _fine_chunk(rng):
    from scipy.ndimage import gaussian_filter as _gf
    i0, i1 = rng
    rfd = os.open(dg, os.O_RDONLY)
    win = np.outer(np.hanning(134), np.hanning(226))
    def feat(i):
        x = np.frombuffer(os.pread(rfd, FSZ, 2 * i * FSZ), '<u2').reshape(M.H, M.W)[30:298, 192:1096].astype(np.float64)
        x = x.reshape(134, 2, 226, 4).mean((1, 3)); return (x - _gf(x, 4)) * win
    out = []; prev = None
    for i in range(i0, i1):
        if not (same[i] or same[i + 1]):
            prev = None; out.append((i, -1.0)); continue
        f = feat(i)
        if same[i]:
            if prev is None: prev = feat(i - 1)
            ea, eb = np.sqrt((f ** 2).mean()), np.sqrt((prev ** 2).mean())
            if min(ea, eb) < FINE_EMIN:
                out.append((i, 2.0))                      # flat/black: keep the coarse link
            else:
                xc = np.fft.irfft2(np.fft.rfft2(f) * np.conj(np.fft.rfft2(prev)), f.shape)
                xc = np.concatenate([xc[-6:], xc[:7]], 0); xc = np.concatenate([xc[:, -12:], xc[:, :13]], 1)
                out.append((i, xc.max() / (np.sqrt((f ** 2).sum() * (prev ** 2).sum()) + 1e-9)))
        else:
            out.append((i, -1.0))
        prev = f
    os.close(rfd)
    return out
if FINE_MIN > 0:
    with ProcessPoolExecutor(max_workers=8, mp_context=_mp.get_context('fork')) as ex:
        for r in ex.map(_fine_chunk, [(k, min(nfr, k + 256)) for k in range(0, nfr, 256)]):
            for i, v in r: FINE[i] = v
    cut = same & (FINE >= 0) & (FINE < FINE_MIN)
    log('fine-detail link test (ZIP_FINE %.2f, flat < %.0f exempt): %d of %d coarse links cut, %d kept as flat'
        % (FINE_MIN, FINE_EMIN, cut.sum(), same.sum(), (same & (FINE == 2.0)).sum()))
    same = same & ~cut
    np.save(os.path.join(a.scratch, 'fine.npy'), FINE)
    FINE_CUT = int(cut.sum())
else:
    FINE_CUT = 0
video = np.array([all(same[k] for k in range(max(0, i - 2) + 1, min(nfr - 1, i + 2) + 1))
                  and min(nfr - 1, i + 2) - max(0, i - 2) == 4 for i in range(nfr)])
np.save(os.path.join(a.scratch, 'video.npy'), video)
log(f'de-smear+herringbone+classify {time.time()-T0:.1f} s'); T1 = time.time()
log(f'{nfr} frames: {video.sum()} video (transform3d), {nfr - video.sum()} stills (model decoder)')

# ---- 3: starting decodes (streamed) -----------------------------------------------
# 2026-10-27: 4 decoder threads (PALDEC_CHROMA_T; 0 = ld-chroma-decoder's default, one per CPU): output identical
# to the default, peak memory per decoder ~14 -> ~1.8 GB on 1500 frames, still 16-26 fps against the solver's ~1.5.
# Added here, not in the p2 / p3 lines: nd_gpuwrap.py (ND_FORCE_T3) patches the p3 line by its exact text.
CHROMA_T = ['-t', os.environ.get('PALDEC_CHROMA_T', '4')] if os.environ.get('PALDEC_CHROMA_T', '4') != '0' else []
class Chroma:
    """One ld-chroma-decoder y4m stream over the de-smeared TBC.  Reading the
    TBC over drvfs intermittently fails ('Could not read field data', 2026-10-12
    and -13); the stream is then restarted at the frame it stopped on."""
    RETRIES = 8
    def __init__(self, args):
        self.args, self.fails = args, 0
        self.start(0)
        self.header = self.hdr
    def start(self, i):
        self.p = subprocess.Popen(['ld-chroma-decoder'] + self.args + CHROMA_T + ['-s', str(i + 1), '-p', 'y4m', '--input-json', dg + '.json', dg, '-'],
                                  stdout=subprocess.PIPE, stderr=open(os.path.join(a.scratch, 'chroma_%s.log' % self.args[1]), 'a'), bufsize=FB * 4)
        h = b''
        while not h.endswith(b'\n'):
            c = self.p.stdout.read(1)
            if not c:
                break
            h += c
        self.hdr, self.i = h, i
    def frame(self, i):
        assert i == self.i
        while True:
            m = self.p.stdout.read(6)
            b = self.p.stdout.read(FB) if m == b'FRAME\n' else b''
            if len(b) == FB:
                self.i += 1
                return np.frombuffer(b, '<u2').reshape(3, OH, OW).astype(np.float32)
            self.fails += 1
            if self.fails > self.RETRIES:
                raise SystemExit('chroma decoder stream failed %d times (see %s/chroma_*.log)' % (self.fails, a.scratch))
            log(f'{self.args[1]} stream ended early at frame {i}; restarting there (retry {self.fails})')
            self.p.kill(); self.p.wait()
            time.sleep(10)
            self.start(i)
p2 = Chroma(['-f', 'transform2d', '--transform-threshold', '0.7'])
p3 = Chroma(['-f', 'transform3d']) if video.any() else None
H2 = p2.header

_lp = None
if a.luma_lpf:
    _f = np.fft.rfftfreq(OW, 1 / 17.734475)
    _lp = (0.5 + 0.5 * np.cos(np.pi * np.clip((_f - (a.luma_lpf - 0.4)) / 0.8, 0, 1))).astype(np.float32)
# v6_3 (2026-10-31): luma equaliser from the TBC's own multiburst VITS (lumaeq.py): the deghost inverse lifts luma
# above ~1 MHz by up to 1.18 while the real record/playback chain is much flatter than its smear model (South after
# the inverse: 1.06 at 1 MHz, 1.08 at 2, 1.17 at 4).  1 / the measured response, never a boost; decoded luma only.
# PALDEC_LUMA_EQ=0: off (= v6_2sat).  Measured over the whole TBC (not the --start/--frames chunk) so chunks agree.
LUMA_EQ = os.environ.get('PALDEC_LUMA_EQ', '1') == '1'
if LUMA_EQ:
    import lumaeq
    _resp, _neq, _loc = lumaeq.measure(a.tbc, channel_inverse)
    if _resp is None:
        log('luma EQ: no multiburst found (%d frames) - off' % _neq)
    else:
        _eq = lumaeq.eq_curve(_resp, OW)
        _lp = _eq if _lp is None else (_lp * _eq).astype(np.float32)
        _fq = np.fft.rfftfreq(OW, 1 / 17.734475)
        log('luma EQ: multiburst field %d line %d, %d frames; response after deghost %s at %s MHz -> gain %s at 1/2/3/4/5 MHz'
            % (_loc[0], _loc[1] + 1, _neq, np.round(_resp, 3).tolist(), list(lumaeq.F0),
               [round(float(_eq[np.argmin(abs(_fq - q))]), 3) for q in (1, 2, 3, 4, 5)]))

# ---- 4+5: per window of frames, pick a decoder per frame ----------------------
#   film / held stills (same scene as a neighbour, both fields one instant):
#       solve_film over the run (temporal prior, motion-compensated)
#   isolated stills: solve (frame mode), batched
#   interlaced video (fields differ in time): transform3d for now
from combscore import comb_score, comb_lowsat, match_score, classify_interlaced, C_T as COMB_T
# PALDEC_CLASSIFY=comb: the old rule (comb_score > 0.58 alone); default 'match' (combscore.classify_interlaced)
CLASSIFY = os.environ.get('PALDEC_CLASSIFY', 'match')
FILM2_SPLIT = float(os.environ.get('PALDEC_FILM2_SPLIT', '3.0'))
FILM_STATIC_M = float(os.environ.get('PALDEC_FILM_STATIC_M', '0.2'))   # held stills 0.05-0.2 (bars 0.5), motion >= 0.7
NW_FILM_LINEART = os.environ.get('PALDEC_NW_FILM_LINEART', '1') == '1'
HELD_LOCAL_VIDEO = os.environ.get('PALDEC_HELD_LOCAL_VIDEO', '1') == '1'   # local-motion runs -> per-field video solve
LOCAL_HELD = os.environ.get('PALDEC_LOCAL_HELD', '1') == '1'               # ... then held-averaged (static background)
LOCAL_HELD_T = float(os.environ.get('PALDEC_LOCAL_HELD_T', 1.0))            # its outlier threshold x this
WIN, RUNCH, VCHUNK = 64, 8, 8      # VCHUNK 8 (vidsplice: 12): W5 + Hanover at 10 frames peaks ~2.6 GB
out = sys.stdout.buffer if a.out == '-' else open(a.out, 'wb')
out.write(H2)
# explicit positional reads, not np.memmap: on drvfs (WSL) memmap page faults go
# through 9P and, alongside two ld-chroma-decoders reading the same file, the
# decoders intermittently failed with 'Could not read field data' (2026-10-12)
_dgfd = os.open(dg, os.O_RDONLY)
def tbc_fields(f0, n):
    b = os.pread(_dgfd, n * FSZ, f0 * FSZ)
    return np.frombuffer(b, '<u2').reshape(n, M.H, M.W)
stats = dict(still=0, film=0, interlaced=0, fine_cut=FINE_CUT, sat_fallback=0)
T_SOLVE = [0.0]


def setups(idx):
    S = [O.frame_setup(tbc_fields(2 * i, 2).astype(np.float32), 0) for i in idx]
    return (torch.stack([x[k] for x in S]) for k in range(3))


# v4: saturation guard for the inline video solve.  Its temporal chroma priors drain the colour of
# fast field-to-field changes (map zoom 47999: red roads 33% of t2d; test-signal cut 3025: garbled).
# Ratio = mean smoothed |C| of the solve / of its transform3d init over pixels where the init's
# |C| > 6 (8-bit); real video 0.92-0.93.  Below SAT_MIN the frame keeps transform3d.  Every
# video-solved frame is logged to <scratch>/satratio.txt (frame ratio fellback).  ZIP_SATGUARD=0 disables.
SAT_MIN = float(os.environ.get('ZIP_SATGUARD', 0.6))   # 10-28: 0.88 -> 0.6: real video solves read 0.6-0.87 on NationalB / City_Culture (low-chroma scenes, t3d cross-colour), so 0.88 sent ~all of them back to transform3d (motion combing x1.5); the drained map zoom read 0.33
SATLOG = open(os.path.join(a.scratch, 'satratio.txt'), 'a')
def _satratio(x, init):
    from scipy.ndimage import uniform_filter as _uf
    def mag(z): return np.hypot(_uf(z[1] - 32768, 5), _uf(z[2] - 32768, 5))
    m0 = mag(init); m = m0 > SAT_C * 256
    return float(mag(x)[m].mean() / m0[m].mean()) if m.sum() > 1000 else 1.0
SAT_C = float(os.environ.get('PALDEC_SATGUARD_C', 6))   # init chroma (8-bit) that the guard tests
GHOST_MAX = float(os.environ.get('PALDEC_GHOST_MAX', 0.3))    # video-solve lean toward its neighbours; 0 = off
GHOST_GAIN = float(os.environ.get('PALDEC_GHOST_GAIN', 0.10))  # ... kept anyway when t3d would comb > this much more
GH_S2 = (O.noise_sigma(_psd) * float(os.environ.get('PALDEC_GHOST_SK', 1.0))) ** 2      # t2d luma noise variance (approx.)
def _ghost_coef(Yd, Yk, Yp, Yn):
    """lean of the solve toward its neighbours, less the part a pure denoise of Yk's own noise gives
    (E[(out - Yk) N] = sigma^2 per pixel when out removes Yk's noise: sigma^2 / mean(N^2))"""
    N = 0.5 * (Yp.astype(np.float64) + Yn) - Yk
    n2 = float((N * N).mean())
    if np.sqrt(n2) < 1000:                       # neighbours ~ this frame (still / noise only): no lean to measure
        return 0.0
    g = float(((Yd.astype(np.float64) - Yk) * N).mean() / n2)
    return g - GH_S2 / n2


# v6_2 (2026-10-30): frames whose output is ld-chroma-decoder's own picture (interlaced frames with --video 0, video
# solves sent back by the saturation / ghost guards) took its chroma scale on the DE-SMEARED TBC: the deghost inverse
# G lifts 4.43 MHz by 1.18 (deghost.py), so on Community South's EBU bars it reads chroma 1.094 (U 1.105, V 1.084)
# against the model solves' 1.003 (U 1.020, V 0.989; GU/GV of model_params.json are set on these bars).  A guard
# fallback frame was therefore ~9% more saturated than the solved frames around it.  Their chroma is scaled to the
# model's: U x0.923, V x0.913 = solve / transform3d on the same de-smeared bars (decoder-to-decoder, so independent of
# the disc channel).  PALDEC_T3_CGAIN=1,1 restores v6_1.
T3_CGAIN = [float(v) for v in os.environ.get('PALDEC_T3_CGAIN', '0.923,0.913').split(',')]
T3SET = set()        # frames (this window) whose output chroma is transform2d / transform3d's


def t3_chroma(x):
    if T3_CGAIN == [1.0, 1.0]:
        return x
    x = np.array(x, np.float32, copy=True)
    x[1] = (x[1] - 32768.) * T3_CGAIN[0] + 32768.; x[2] = (x[2] - 32768.) * T3_CGAIN[1] + 32768.
    return x


def decode_window(i0, F2, F3):
    """F2/F3: dict frame -> init (transform2d / transform3d).  Returns dict frame -> output."""
    T3SET.clear()
    idx = sorted(F2)
    res = {}
    held = set(); fbf = set()
    if NW_ADAPT:
        REGION_W.clear(); REGION_W.update({i: region_weight(F2[i][0]) for i in idx})
        REGION_WF.clear(); REGION_WF.update({i: region_weight(F2[i][0], NW_FG0, NW_FG1) for i in idx})
        stats['nw_flat_sum'] = stats.get('nw_flat_sum', 0) + round(float(sum((w < 0.99).mean() for w in REGION_W.values())), 1)
    cs = np.array([comb_score(F2[i][0]) for i in idx])   # combscore: static <=0.50
    if CLASSIFY == 'comb':
        inter = {i: bool(c > COMB_T) for i, c in zip(idx, cs)}
    else:
        # static frames that only LOOK combed (saturated colour, menus, text pages, bars) match their same-scene
        # neighbour to noise; real interlaced motion does not (combscore.classify_interlaced)
        link = np.array([k > 0 and bool(same[i]) and idx[k - 1] == i - 1 for k, i in enumerate(idx)])
        mm = np.full(len(idx), np.inf)
        for k, i in enumerate(idx):
            if cs[k] > COMB_T:
                if link[k]: mm[k] = min(mm[k], match_score(F2[i][0], F2[i - 1][0]))
                if k + 1 < len(idx) and link[k + 1]: mm[k] = min(mm[k], match_score(F2[i][0], F2[i + 1][0]))
        cl = np.array([comb_lowsat(F2[i]) if cs[k] > COMB_T else np.nan for k, i in enumerate(idx)])
        iv = classify_interlaced(cs, mm, link, cl)
        inter = {i: bool(v) for i, v in zip(idx, iv)}
        stats['comb_still'] = stats.get('comb_still', 0) + int(((cs > COMB_T) & ~iv).sum())
    # runs of single-instant frames linked by same-scene correlation
    runs, cur = [], []
    vids = []
    for i in idx:
        if inter[i]:
            if cur: runs.append(cur); cur = []
            res[i] = F3[i] if F3 else F2[i]; stats['interlaced'] += 1; T3SET.add(i)
            vids.append(i)
            continue
        if cur and same[i]:
            cur.append(i)
        else:
            if cur: runs.append(cur)
            cur = [i]
    if cur: runs.append(cur)
    # 2026-10-27 (herring integration): a 2-frame film run of two DIFFERENT pictures (consecutive satellite photos of
    # similar layout pass the coarse and fine link tests: North 16395/16396, match 4.8) is split into two stills:
    # solve_film's temporal prior mixes them and leaves the subcarrier in luma (hatch x10, lost saturation).  Film
    # frames of real motion match 1.2-2.3; PALDEC_FILM2_SPLIT=0 disables.
    if FILM2_SPLIT > 0:
        sp = []
        for r in runs:
            if len(r) == 2 and match_score(F2[r[1]][0], F2[r[0]][0]) > FILM2_SPLIT:
                sp += [[r[0]], [r[1]]]; stats['film2_split'] = stats.get('film2_split', 0) + 1
            else:
                sp.append(r)
        runs = sp
    singles = [r[0] for r in runs if len(r) == 1]
    t = time.time()
    for k in range(0, len(singles), BATCH):
        b = singles[k:k + BATCH]
        C, cu, cv = setups(b)
        o = O.solve(C, cu, cv, np.stack([F2[i] for i in b]), noise_w=NOISE_W, dmask=dmask_solve(b), **PARAMS)
        for i, x in zip(b, o): res[i] = x
        stats['still'] += len(b)
    local_runs = []; film_res = {}
    for r in (r for r in runs if len(r) > 1):
        for k in range(0, len(r), RUNCH):
            core = r[k:k + RUNCH]
            ctx = r[max(0, k - 1):min(len(r), k + RUNCH + 1)]          # one frame of overlap each side
            C, cu, cv = setups(ctx)
            I = np.stack([(F3 or F2)[i] if (F3 and video[i]) else F2[i] for i in ctx])
            fl = O.estimate_flow_gpu(I[:, 0], lag=1)
            # region-adaptive weights for static runs only: on MOVING film the p75 detail weight keeps line structure
            # that reads as combing (quickcheck suite EcoDisc / BGB / NationalB motion: 1.25-1.33 x v6_0), while static
            # film runs (maps, text, bars that held_static did not take) need it for their detail (NationalA / BGB maps
            # rise 1.10-1.12 x with the min weights).  Static: median frame-to-frame match_score of the inits < FILM_STATIC_M.
            st = NW_ADAPT and np.median([match_score(F2[ctx[q + 1]][0], F2[ctx[q]][0]) for q in range(len(ctx) - 1)]) < FILM_STATIC_M
            stats['film_static' if st else 'film_moving'] = stats.get('film_static' if st else 'film_moving', 0) + len(core)
            # moving film: the p75 detail weight only on strong line-art edges (maps / text / graphics: line contrast
            # far above any natural texture), elsewhere the min level (natural moving texture read as combing)
            fw = NW_ADAPT and NW_FILM_LINEART
            o = O.solve_film(C, cu, cv, I, fl, noise_w=NOISE_W if (st or fw) else NOISE_WV, hanover=PARAMS['hanover'],
                             dmask=dmask_solve(ctx) if st else dmask_solve(ctx, film=True) if fw else dmask(ctx), **FILM)
            for i, x in zip(ctx, o):
                if i in core: res[i] = x
            stats['film'] += len(core)
        if O.HELD and len(r) >= O.HELD_MIN:
            # held still: one shared latent over the whole static run
            # (within this window), per-frame outlier blocks keep their own solution
            hs = O.held_static(np.stack([res[i][0] for i in r]))
            print(f"[newdecode] held? {r[0]}-{r[-1]} flow {O.HELD_DEBUG['flow']:.3f} drift {O.HELD_DEBUG['drift']:.2f} movfrac {O.HELD_DEBUG['movfrac']:.3f} -> {'held' if hs else 'moving'}", file=sys.stderr)
            if hs:
                C, cu, cv = setups(r)
                o = O.solve_held(C, cu, cv, np.stack([res[i] for i in r]), NOISE_W, PARAMS, hanover=PARAMS['hanover'], dmask=dmask_solve(r))
                del C, cu, cv
                for i, x in zip(r, o): res[i] = x
                held.update(r)
                stats['held'] = stats.get('held', 0) + len(r)
                stats['held_out'] = stats.get('held_out', 0) + O.HELD_DEBUG['outblocks']
                print(f"[newdecode] held {r[0]}-{r[-1]} outblocks {O.HELD_DEBUG['outblocks']}", file=sys.stderr)
            elif (HELD_LOCAL_VIDEO and a.video and F3 and O.HELD_DEBUG['flow'] < O.HELD_FLOW
                  and O.HELD_MOVFRAC > 0 and O.HELD_DEBUG['movfrac'] >= O.HELD_MOVFRAC):
                # local motion on a still background (NationalB 69442 inset of moving people, talking heads): the
                # frame-level classifier sees a still (comb score of the whole frame) and weaves it, but the moving
                # region is interlaced video -> combing (NationalB_v2 0.254 film / 0.225 held vs v6_0 0.187, which
                # averages the motion away: ghost 0.77).  Solved per field as video: combing 0.199, ghost 0.53.
                # The still background is then averaged over the run from those per-field solutions (solve_held:
                # its outlier blocks, the moving region, keep each frame's own video solution): Countryside talking-
                # head insets, all-video flat grain 183 vs v6_0's averaged 117.
                for i in r:
                    film_res[i] = res[i]; res[i] = F3[i]; vids.append(i); inter[i] = True; T3SET.add(i)
                stats['film'] -= len(r); stats['interlaced'] += len(r)
                stats['local_video'] = stats.get('local_video', 0) + len(r)
                local_runs.append(r)
            else:
                stats['held_moving'] = stats.get('held_moving', 0) + len(r)
    vids.sort()
    if a.video and vids and F3:
        # interlaced video: per-field solve (as vidsplice.py), runs of consecutive frames in
        # chunks of VCHUNK with one context frame each side (inside this window)
        vr, cur = [], []
        for i in vids:
            if cur and i == cur[-1] + 1: cur.append(i)
            else:
                if cur: vr.append(cur)
                cur = [i]
        if cur: vr.append(cur)
        for r in vr:
            # v5: a short run (lone interlaced frames at cuts) has no related neighbour to fill a dropout from;
            # its temporal priors would fill it from a different picture (random-frame test: luma err 16 -> 28)
            vmask = a.dropout_mask and len(r) >= DO_VMIN
            for k in range(0, len(r), VCHUNK):
                core = r[k:k + VCHUNK]
                ks = list(range(max(idx[0], core[0] - 1), min(idx[-1], core[-1] + 1) + 1))
                tb = tbc_fields(2 * ks[0], 2 * len(ks)).astype(np.float32)
                C, cu, cv = O.field_setup(tb, range(len(tb)))
                I = np.array([F3[kk][:, p::2] for kk in ks for p in (0, 1)])
                fl = O.estimate_flow_gpu(I[:, 0], lag=2)
                o = O.solve_fields(C, cu, cv, I, [p for _ in ks for p in (0, 1)], flow=fl, noise_w=NOISE_WV,
                                   hanover=CFG['hanover_video'], lr=1e4, **CFG['video'],
                                   dmask=np.array([fmask(2 * kk + p) for kk in ks for p in (0, 1)]) if vmask else None)
                for c, kk in enumerate(ks):
                    if kk in core:
                        x = np.empty((3, OH, OW), np.float32); x[:, 0::2] = o[2 * c]; x[:, 1::2] = o[2 * c + 1]
                        ratio = _satratio(x, F3[kk]); fb = ratio < SAT_MIN; sat_fb = fb
                        # 10-28 ghost guard: a video solve whose luma leans toward the neighbouring frames (its
                        # temporal priors mixed them in) keeps transform3d, as the 0.88 guard did by accident
                        # (UN_v3 motion ghost 0.45 vs v6_0 -0.03).  Lean as quickcheck's ghost_coef, vs the t2d inits.
                        if not fb and GHOST_MAX > 0 and kk - 1 in F2 and kk + 1 in F2:
                            gc = _ghost_coef(x[0], F2[kk][0], F2[kk - 1][0], F2[kk + 1][0])
                            if os.environ.get('PALDEC_GHOST_LOG'):
                                print('[newdecode] glog %d lean %.3f comb_solve %.3f comb_t3d %.3f comb_t2d %.3f' % (kk, gc,
                                      comb_score(x[0]), comb_score(F3[kk][0]), comb_score(F2[kk][0])), file=sys.stderr)
                            # lesser artefact: transform3d only where the solve leans (ghost) AND buys little against
                            # combing (comb_score of t3d over the solve's < 1 + GHOST_GAIN): UN 24-30 lean 0.45, gain 1-9 %
                            # -> t3d; City_Culture1 lean 0.44, gain 25 % / City_Financial1 0.52, 15 % -> solve
                            if gc > GHOST_MAX and GHOST_GAIN > 0:
                                cs_s = comb_score(x[0]); cs_t = comb_score(F3[kk][0])
                                if cs_t > cs_s * (1 + GHOST_GAIN): gc = 0.0
                            if gc > GHOST_MAX:
                                fb = True; stats['ghost_fallback'] = stats.get('ghost_fallback', 0) + 1
                                print('[newdecode] ghost guard frame %d lean %.2f -> transform3d' % (kk, gc), file=sys.stderr)
                        # satratio.txt / sat_fallback record the saturation guard only; ghost fallbacks are counted
                        # in stats['ghost_fallback'] (quality DB nd table must not mislabel them)
                        SATLOG.write('%d %.3f %d\n' % (a.start + kk, ratio, sat_fb)); SATLOG.flush()
                        if sat_fb: print('[newdecode] satguard frame %d ratio %.3f -> transform3d' % (kk, ratio), file=sys.stderr)
                        if sat_fb: stats['sat_fallback'] += 1
                        if fb:
                            if vmask and (2 * kk in DO or 2 * kk + 1 in DO):
                                # v5: transform3d's dropout pixels are line copies; keep the masked solve's luma there
                                # (its chroma is what failed the guard: synthetic test luma err 9.3 -> 4.4, chroma 1.7 -> 2.0)
                                hole = dmask([kk])[0] < 0.5
                                x = np.concatenate([np.where(hole, x[0], F3[kk][0])[None], F3[kk][1:]])
                                stats['fb_filled'] = stats.get('fb_filled', 0) + 1; fbf.add(kk)
                            else:
                                x = F3[kk]
                            T3SET.add(kk)
                        else:
                            T3SET.discard(kk)
                        res[kk] = x
                stats['video_solved'] = stats.get('video_solved', 0) + len(core)
    if LOCAL_HELD:
        for r in local_runs:
            if all(res[i] is F3[i] for i in r):          # no video solve ran for this run
                continue
            C, cu, cv = setups(r)
            _t = O.HELD_T; O.HELD_T = _t * LOCAL_HELD_T      # per-field solutions are noisier than film ones
            try:
                # the static background averaged from the FILM solutions (independent per frame: the video solve's
                # temporal priors correlate its frames, Countryside 27826 grain 155 vs 109); the moving blocks
                # (outliers) take the per-field video solutions
                o = O.solve_held(C, cu, cv, np.stack([film_res[i] for i in r]), NOISE_W, PARAMS, hanover=PARAMS['hanover'],
                                 dmask=dmask(r), own=np.stack([res[i] for i in r]))
            finally:
                O.HELD_T = _t
            del C, cu, cv
            for i, x in zip(r, o): res[i] = x; T3SET.discard(i)
            held.update(r)
            stats['local_held'] = stats.get('local_held', 0) + len(r)
            print(f"[newdecode] local-motion held {r[0]}-{r[-1]} outblocks {O.HELD_DEBUG['outblocks']}", file=sys.stderr)
    if a.dropout_mask and DOI in ('replace', 'mix'):
        for i in idx:
            if i in held or not (2 * i in DO or 2 * i + 1 in DO):
                continue
            m = doi_mask(i)
            if not m.any():
                continue
            x = np.asarray(res[i], np.float32)
            y = doi_fill(x, m, [-8, -6, -4, -2, 2, 4, 6, 8] if inter[i] else [-4, -3, -2, -1, 1, 2, 3, 4])
            if i in fbf and DOI_FB_LUMA:      # v6: a guard-fallback frame keeps the masked solve's luma in its holes
                y[0] = x[0]
            res[i] = y if DOI == 'replace' else 0.5 * (x + y)
            stats['doi'] = stats.get('doi', 0) + 1
    if a.dropout_mask and DOT > 0:
        src = {i: np.asarray(res[i], np.float32) for i in idx}
        msk = {i: doi_mask(i) for i in idx if (2 * i in DO or 2 * i + 1 in DO)}
        for i, m in msk.items():
            if not m.any():
                continue
            x = src[i]; ssum = np.zeros_like(x); cnt = np.zeros(m.shape, np.float32)
            for jn in (i - 1, i + 1):
                if jn not in src:
                    continue
                y = src[jn]; mj = msk.get(jn, np.zeros_like(m))
                w = (~m & ~mj).astype(np.float32)
                dy = _boxmean(np.abs(x[0] - y[0]), w) / 256
                dc = _boxmean(0.5 * (np.abs(x[1] - y[1]) + np.abs(x[2] - y[2])), w) / 256
                st = (dy < DOT_T) & (dc < DOT_T) & m & ~mj
                ssum[:, st] += y[:, st]; cnt[st] += 1
            sel = cnt > 0
            if sel.any():
                z = x.copy(); z[:, sel] = ssum[:, sel] / cnt[sel]; res[i] = z
                stats['dot_px'] = stats.get('dot_px', 0) + int(sel.sum())
    T_SOLVE[0] += time.time() - t
    if os.environ.get('ZIP_DUMP'):          # debug: per-frame class log
        cls = {i: ('inter' if inter[i] else 'single') for i in idx}
        for r in runs:
            if len(r) > 1:
                for i in r:
                    if not inter[i]: cls[i] = 'film%d' % len(r)
        with open(os.environ['ZIP_DUMP'], 'a') as fh:
            for i in idx: fh.write('%d %s video=%d same=%d fine=%.3f\n' % (a.start + i, cls[i], video[i], same[i], FINE[i]))
    return res


# v6: window ends at a run boundary (a frame not linked to its predecessor) where possible: v5's fixed 64-frame
# windows split held / film runs, leaving short fragments that are solved as singles or short runs
# (South: 8 single-frame fragments).  A run crossing w0+WIN is taken whole when it ends by w0+WIN_MAX, else
# the window ends where it starts (if that leaves >= WIN/4 frames), else it is split at w0+WIN (pieces >= 16).
# PALDEC_WIN_MAX=64 restores v5.
# v6 agent 10-27: 80 (a 96-frame held window peaked at 5.3 GB GPU, 64: 3.2; 80 still leaves no piece shorter than 16)
WIN_MAX = int(os.environ.get('PALDEC_WIN_MAX', 80))
def window_end(w0):
    e = min(nfr, w0 + WIN)
    if e == nfr or not same[e] or WIN_MAX <= WIN:
        return e
    t = next((b for b in range(e + 1, min(nfr, w0 + WIN_MAX) + 1) if b == nfr or not same[b]), None)
    if t is not None:
        return t
    s0 = next((b for b in range(e - 1, w0 + WIN // 4 - 1, -1) if not same[b]), None)
    return s0 if s0 is not None else e
w0 = 0
while w0 < nfr:
    w1 = window_end(w0)
    F2 = {}; F3 = {} if p3 else None
    for i in range(w0, w1):
        F2[i] = p2.frame(i)
        if p3: F3[i] = p3.frame(i)
    res = decode_window(w0, F2, F3)
    for i in range(w0, w1):
        x = res[i]
        if i in T3SET:          # v6_2: transform3d / 2d output to the model's chroma scale
            x = t3_chroma(x)
        if _lp is not None:
            x = np.array(x, np.float32, copy=True); x[0] = np.fft.irfft(np.fft.rfft(x[0], axis=1) * _lp, OW, axis=1)
        out.write(b'FRAME\n'); out.write(np.clip(np.round(x), 0, 65535).astype('<u2').tobytes())
    log(f'frame {w1}/{nfr}  {stats}')
    w0 = w1

out.flush()
os.close(_dgfd)
for _x in ('', '.json', '.stamp', '.coarse.npy'):
    os.remove(dg + _x)
log(f'done: total {time.time()-T0:.1f} s, solves {T_SOLVE[0]:.1f} s; frames {stats}')
