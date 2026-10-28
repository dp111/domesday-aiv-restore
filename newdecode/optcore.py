# SPDX-License-Identifier: GPL-3.0-or-later
"""Core of the model-based PAL decoder (see optdec.py)."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palmodel as M
OW, OH, X0, LINE0 = 928, 576, 182, 22
FS = 17.734475
dev = 'cuda'


def _dm(dmask, C):
    """per-pixel data weight (v5, 2026-10-19): a tensor shaped like C with 0 on dropout samples
    (single-copy discs), so the spatial / temporal priors - and in a held run the other frames -
    fill them; a scalar 1 when no mask is given (output identical to v4)"""
    if dmask is None:
        return torch.ones((), device=dev)
    return torch.as_tensor(np.asarray(dmask, np.float32), device=dev)


def frame_setup(tbc, fi):
    comp = np.zeros((OH, OW), np.float32); CU = np.zeros_like(comp); CV = np.zeros_like(comp)
    for par in (0, 1):
        fld = 2 * fi + par
        b0, sp = M.field_burst(tbc[fld])
        rows = np.arange(par, OH, 2)
        cu, cv = M.carriers(b0, sp, nlines=len(rows), width=OW, x0=X0, line0=LINE0)
        comp[rows] = tbc[fld, LINE0:LINE0 + len(rows), X0:X0 + OW]
        CU[rows] = cu; CV[rows] = cv
    t = lambda x: torch.tensor(x, device=dev)
    return t(comp), t(CU), t(CV)


_hp = {}
def hf_mask(mhz):
    if mhz not in _hp:
        f = np.fft.rfftfreq(OW, 1 / FS)
        _hp[mhz] = torch.tensor((f > mhz).astype(np.float32), device=dev)
    return _hp[mhz]


def _frame_loss_w(y, cb, cr, C, cu, cv, Dm, wf, AY, B0, lam_y, eps_y, lam_cx, lam_cy, eps_c):
    """As _frame_loss, with the data term weighted per horizontal frequency by
    wf ~ 1 / noise power (mean 1): the decoder trusts each band of the
    composite in proportion to how little of it is noise.  On the discs the
    5.5-7 MHz band is almost all disc noise, so luma there is left to the
    prior instead of being copied from the noise."""
    Y, Cb, Cr = y * 256, cb * 256, cr * 256
    r = (AY * Y + B0 + Cb * cu + Cr * cv - C) * Dm
    R = torch.fft.rfft(r, dim=-1)
    loss = ((R.real ** 2 + R.imag ** 2) * wf).sum() * (2.0 / r.shape[-1])
    gx = Y[..., :, 1:] - Y[..., :, :-1]; gy = Y[..., 1:, :] - Y[..., :-1, :]
    loss = loss + lam_y * eps_y * (torch.sqrt(gx ** 2 + eps_y ** 2).sum() + torch.sqrt(gy ** 2 + eps_y ** 2).sum())
    for c in (Cb, Cr):
        dx = c[..., :, 1:] - c[..., :, :-1]; dy = c[..., 1:, :] - c[..., :-1, :]
        loss = loss + eps_c * (lam_cx * torch.sqrt(dx ** 2 + eps_c ** 2).sum() + lam_cy * torch.sqrt(dy ** 2 + eps_c ** 2).sum())
    return loss / 65536.0


def noise_weights(psd, floor=0.05, ceil=None, norm='mean'):
    """Data-term weights from a noise PSD on the 928-sample row bins: 1/P,
    normalised to mean 1, floored so no band is ignored outright.  ceil: optional
    cap; the near-noiseless >7.5 MHz bins otherwise get weights ~7 against ~0.02
    in band, a stiff direction that leaves L-BFGS under-converged at 60 iterations."""
    w = 1.0 / np.maximum(psd, 1e-9)
    if norm == 'inband':
        # v6: scaled so the quietest picture band (0-5 MHz, 0.5 MHz smoothed) sits at the cap; v5 divided by the
        # mean over all bins, which the near-noiseless bins above ~7.5 MHz (TBC low-pass) dominate, so the
        # in-band weights fell to the floor or rose to the cap depending on how quiet that band happened to be
        # (fq bench: all at the floor; real South clips: mostly at the cap, i.e. 4x the bench's data weight)
        f = np.fft.rfftfreq(2 * (len(psd) - 1), 1 / FS)
        sm = np.convolve(np.pad(psd, 13, mode='edge'), np.ones(27) / 27, 'valid')
        # PALDEC_NW_REF: the in-band level that sits at the cap - 'min' (the quietest band) or pNN, the NN-th
        # percentile of the 0.2-5 MHz spectrum (bins quieter than it are capped too)
        ib = sm[(f > 0.2) & (f < 5.0)]
        ref = float(np.min(ib)) if NW_REF == 'min' else float(np.percentile(ib, float(NW_REF[1:])))
        w = w * ref * (ceil or 1.0)
    else:
        w = w / w.mean()
    w = np.maximum(w, floor)
    if ceil:
        w = np.minimum(w, ceil)
    # v6: the losses sum |rfft|^2 * w * 2/N over the half spectrum, which counts the DC and Nyquist bins
    # twice (they have no mirror bin); halve their weights (PALDEC_NW_EDGES=0: v5)
    if NW_EDGES:
        w = w.copy(); w[0] *= 0.5; w[-1] *= 0.5          # rows are 928 samples (even): the last bin is Nyquist
    return w.astype(np.float32)


def noise_sigma(psd):
    """rms noise (16-bit units) of a noise_psd() spectrum (v6: DC and Nyquist bins counted once)"""
    if not NW_EDGES:
        return float(np.sqrt(np.sum(psd) * 2 / 928 ** 2))
    return float(np.sqrt((2 * np.sum(psd) - psd[0] - psd[-1]) / 928 ** 2))


# ---- decoder configuration (combined candidate, 2026-10-14) --------------------
# Every change against the 2026-10-13 production decoder is switchable by env var:
#   PALDEC_NW=rec|cap|prod  noise weights: rec = floor 0.00625, cap 0.025 (stills bench);
#                           cap = floor 0.05, cap 0.2; prod = floor 0.05, no cap
#   PALDEC_PRIORS=rec|prod  spatial/temporal prior strengths: rec = the stills-bench
#                           values scaled by the stack's noise level k = sigma/928
#   PALDEC_HANOVER=LAM      stage-2 Hanover prior weight (0 = off)
#   PALDEC_W5=1|0           video solver motion terms (W5), PALDEC_VITERS iterations (90)
#   PALDEC_LUMA_LPF=MHZ     output luma roll-off (newdecode/vidsplice default)
#   PALDEC_PARAMS=FILE      carrier calibration json (palmodel; default model_params.json)
#   further knobs: PALDEC_LAMY_SCALE (luma prior x, grain vs detail), PALDEC_REC_S / PALDEC_REC_ST
#   (film/video spatial / temporal prior scale, 0.15 / 0.4), PALDEC_HANOVER_VIDEO (field
#   Hanover stage, default 64 since v4), PALDEC_HANOVER_G0 / _MASKN (stage-2 edge mask, 12 / 9),
#   PALDEC_HCOMPILE (compile the stage-2 prior, 1).  Production behaviour of 2026-10-13:
#   PALDEC_PARAMS=<old json> PALDEC_NW=prod PALDEC_PRIORS=prod PALDEC_HANOVER=0 PALDEC_W5=0, and for v6 also
#   PALDEC_NOISE=vbi PALDEC_NW_EDGES=0 (see README_dropout.txt for the other v6 switches)
NW_MODE = os.environ.get('PALDEC_NW', 'rec')
NW_EDGES = os.environ.get('PALDEC_NW_EDGES', '1') == '1'
NW_REF = os.environ.get('PALDEC_NW_REF', 'min')   # v6 agent 10-27: p75 (sharper maps, +15% flat grain) vs min (v6_0); see integration notes
NW_NORM = os.environ.get('PALDEC_NW_NORM', 'inband')      # v6: inband | mean (v5)
KSCALE_C = float(os.environ.get('PALDEC_KSCALE_C', 1.0))   # v6: extra scale on the chroma priors only
KSCALE = float(os.environ.get('PALDEC_KSCALE', 1.0))     # v6: prior scale k x this (tuning on the corrected bench)
PRIOR_MODE = os.environ.get('PALDEC_PRIORS', 'rec')
REC_S = float(os.environ.get('PALDEC_REC_S', 0.15))     # rec priors / production priors (film and video solvers)
VIDEO_W5 = os.environ.get('PALDEC_W5', '1') == '1'
REC_ST = float(os.environ.get('PALDEC_REC_ST', 0.4))     # same for their temporal priors (film + video benches: 0.4 > 0.15, 1)


def decoder_config(psd):
    """-> dict(noise_w, k, still=..., film=..., video=..., hanover=..., luma_lpf=...)
    for a stack with noise spectrum psd (noisepsd.noise_psd)."""
    k = noise_sigma(psd) / 928.0 * KSCALE
    if NW_MODE == 'rec':
        nw = noise_weights(psd, floor=0.00625, ceil=0.025, norm=NW_NORM)   # v6: PALDEC_NW=prod / cap keep the v5 mean
    elif NW_MODE == 'cap':          # minimal step: production floor, weights capped at 0.2
        nw = noise_weights(psd, ceil=0.2)
    else:
        nw = noise_weights(psd)
    han = float(os.environ.get('PALDEC_HANOVER', HANOVER_DEFAULT))
    if PRIOR_MODE == 'rec':
        ly = float(os.environ.get('PALDEC_LAMY_SCALE', 1.0))     # luma prior knob (grain vs detail)
        still = dict(lam_y=0.006 * k * ly, eps_y=1200., lam_cx=0.045 * k, lam_cy=0.018 * k, eps_c=300.)
        s = REC_S * k
        st = REC_ST * k
        film = dict(lam_y=0.04 * s * ly, eps_y=1200., lam_cx=0.3 * s, lam_cy=0.1 * s, eps_c=300.,
                    lam_t=0.3 * st, eps_t=300., lam_tc=0.3 * st)
        video = dict(lam_y=0.01 * s, eps_y=2000., lam_cx=0.3 * s, lam_cy=0.1 * s, eps_c=300.,
                     lam_t=0.2 * st, eps_t=300., lam_tc=0.3 * st, lam_x=0.0)
        w5 = dict(lam_xm=0.1 * st, lam_xmc=0.1 * st, hp_pow=4., hp_pow1=2., lam_t4=0.1 * st)
        hs = float(os.environ.get('PALDEC_HANOVER_SCALE', REC_S * k))   # Hanover prior vs data, as the priors
    else:
        still = dict(lam_y=0.04, eps_y=1200., lam_cx=0.3, lam_cy=0.1, eps_c=300.)
        film = dict(lam_y=0.04, eps_y=1200., lam_cx=0.3, lam_cy=0.1, eps_c=300., lam_t=0.3, eps_t=300., lam_tc=0.3)
        video = dict(lam_y=0.01, eps_y=2000., lam_cx=0.3, lam_cy=0.1, eps_c=300., lam_t=0.2, eps_t=300., lam_tc=0.3, lam_x=0.0)
        w5 = dict(lam_xm=0.1, lam_xmc=0.1, hp_pow=4., hp_pow1=2., lam_t4=0.1)
        hs = 1.0
    hv = float(os.environ.get('PALDEC_HANOVER_VIDEO', 64))  # v4: on, as stills (bars 3028-3031 zipper: same-field chroma alternation 7.8-9.7 -> 0.5-0.7); video bench -0.1 dB chroma, +35% video-solve time
    if VIDEO_W5:                     # 'W5' motion terms for solve_fields
        video.update(w5)
    # W5 needs 90 iterations with the production weights; with the rec weights 60 = 90 (converged)
    video['iters'] = int(os.environ.get('PALDEC_VITERS', 60 if NW_MODE == 'rec' else 90 if VIDEO_W5 else 60))
    if KSCALE_C != 1.0:              # v6 tuning knob: chroma priors (spatial, temporal, Hanover) x this
        for d in (still, film, video):
            for q in ('lam_cx', 'lam_cy', 'lam_tc', 'lam_xmc'):
                if q in d:
                    d[q] *= KSCALE_C
        hs *= KSCALE_C
    return dict(noise_w=nw, k=k, still=still, film=film, video=video, hanover=han * hs, hanover_video=hv * hs,
                luma_lpf=float(os.environ.get('PALDEC_LUMA_LPF', LUMA_LPF_DEFAULT)))


HANOVER_DEFAULT = 64.
LUMA_LPF_DEFAULT = float(os.environ.get('PALDEC_LUMA_LPF_DEFAULT', 0.0)) if PRIOR_MODE == 'rec' else 5.2


_frame_loss_wc = None


def _frame_loss(y, cb, cr, C, cu, cv, Dm, AY, B0, lam_y, eps_y, lam_cx, lam_cy, eps_c):
    """Frame-mode objective (no lam_hf term).  Kept free of Python control flow
    on tensors so torch.compile fuses it: 13.6 -> 1.3 ms per loss+gradient for a
    batch of 8 frames on the RTX 3060."""
    Y, Cb, Cr = y * 256, cb * 256, cr * 256
    r = (AY * Y + B0 + Cb * cu + Cr * cv - C) * Dm
    loss = (r ** 2).sum()
    gx = Y[..., :, 1:] - Y[..., :, :-1]; gy = Y[..., 1:, :] - Y[..., :-1, :]
    loss = loss + lam_y * eps_y * (torch.sqrt(gx ** 2 + eps_y ** 2).sum() + torch.sqrt(gy ** 2 + eps_y ** 2).sum())
    for c in (Cb, Cr):
        dx = c[..., :, 1:] - c[..., :, :-1]; dy = c[..., 1:, :] - c[..., :-1, :]
        loss = loss + eps_c * (lam_cx * torch.sqrt(dx ** 2 + eps_c ** 2).sum() + lam_cy * torch.sqrt(dy ** 2 + eps_c ** 2).sum())
    return loss / 65536.0


_frame_loss_c = None
# L-BFGS history: 8 is within 0.1 dB of 20 and ~1.7x faster (memory traffic over
# 12.8M unknowns per batch dominates once the objective is compiled)
HIST = int(os.environ.get('PALDEC_HIST', '8'))
NEVAL = [0]


# ---- Hanover-bar suppression (stage 2) -----------------------------------------
# Same-field line alternation of the chroma is what a PAL-D delay line cancels: noise
# and residual phase error near fsc that a single-line exact model can put into
# alternating chroma (Hanover bars).  After the normal solve, a few more L-BFGS
# iterations add a prior on the SUSTAINED alternation only (field vertical Nyquist
# +- ~1/N, Hann over N field lines), switched off at vertical chroma steps found on
# the stage-1 solution, so edges and thin coloured lines keep their sharpness.
# Starting from the stage-1 solution matters: from the soft transform2d start the
# prior holds vertical colour edges at transform2d softness.
HANOVER_N = 8
# mask threshold (16-bit units) and vertical dilation (field lines) of the stage-2 edge mask
HANOVER_G0 = float(os.environ.get('PALDEC_HANOVER_G0', 12)) * 256.
HANOVER_MASKN = int(os.environ.get('PALDEC_HANOVER_MASKN', 9))   # 9 (Hanover R&D: 5): the 8-line Hann window must not see the step, else ~9% overshoot at vertical colour edges


# South Test Card F 3000-3022 (top bars start on the first picture lines): the prior's zero
# padding reads a saturated flat colour within N/2 lines of a field edge as alternation, and
# the solve cancels it with a real ripple ~10 field lines deep (red Cr field difference
# +-11).  PALDEC_HANOVER_EDGE=1: the prior covers only windows fully inside the field.
HANOVER_EDGE = os.environ.get('PALDEC_HANOVER_EDGE', '1') == '1'


def HANOVER_VALID(h, N, c):
    v = torch.zeros(h, device=c.device, dtype=c.dtype)
    v[N // 2:h - N // 2 + 1] = 1
    return v


def _hanover_mask(c, g0, n=5, fields=False):
    """c: chroma (..., H, W) in 16-bit units.  1 where flat, -> 0 near vertical steps
    of the alternation-free 2-line mean (threshold g0, 16-bit units).
    fields=True: c is already per field (F, 288, W), no frame-row interleave."""
    import torch.nn.functional as Fn
    H, W = c.shape[-2:]
    out = torch.empty_like(c)
    for par in ((0,) if fields else (0, 1)):
        st = 1 if fields else 2
        f = c[..., par::st, :].reshape(-1, (H - par + st - 1) // st, W)
        s2 = 0.5 * (f + torch.cat([f[:, 1:], f[:, -1:]], 1))
        g = torch.zeros_like(f); g[:, 1:-1] = (s2[:, 2:] - s2[:, :-2]).abs()
        g = Fn.avg_pool1d(g.reshape(-1, 1, W), 9, 1, 4, count_include_pad=False).reshape(g.shape)
        g = Fn.max_pool2d(g[:, None], (n, 9), 1, (n // 2, 4))[:, 0]
        out[..., par::st, :] = (1.0 / (1.0 + (g / g0) ** 4)).reshape(c[..., par::st, :].shape)
    return out


def _hanover_prior(c, wm, N=HANOVER_N, fields=False):
    """sum of squared, masked sustained same-field alternation of c (..., H, W)"""
    import torch.nn.functional as Fn
    H, W = c.shape[-2:]
    win = torch.tensor(np.hanning(N + 2)[1:-1] / np.hanning(N + 2)[1:-1].sum(), device=c.device, dtype=c.dtype)[None, None]
    tot = 0.0
    st = 1 if fields else 2
    for par in ((0,) if fields else (0, 1)):
        m = c[..., par::st, :]
        h = m.shape[-2]
        sg = torch.tensor((-1.0) ** np.arange(h), device=c.device, dtype=c.dtype)[:, None]
        q = (m * sg).reshape(-1, h, W).permute(0, 2, 1).reshape(-1, 1, h)
        lp = Fn.conv1d(q, win, padding=N // 2)[..., :h]
        wq = wm[..., par::st, :].reshape(-1, h, W).permute(0, 2, 1).reshape(-1, 1, h)
        if HANOVER_EDGE:     # the zero padding makes a flat colour at the field's top/bottom look like alternation
            wq = wq * HANOVER_VALID(h, N, c)
        tot = tot + ((lp * wq) ** 2).sum()
    return tot


def _hanover_both(cb, cr, wb, wr, lam, fields):
    return lam * (_hanover_prior(cb * 256, wb, fields=fields) + _hanover_prior(cr * 256, wr, fields=fields)) / 65536.0


_hanover_both_c = None
HANOVER_COMPILE = os.environ.get('PALDEC_HCOMPILE', '1') == '1'


def _hanover_stage(y, cb, cr, objective, lam, iters, g0, lr, fields=False):
    """stage 2: re-solve from the current solution with the Hanover prior added"""
    global _hanover_both_c
    g0 = HANOVER_G0 if g0 is None else g0
    with torch.no_grad():
        wb = _hanover_mask(cb * 256, g0, n=HANOVER_MASKN, fields=fields); wr = _hanover_mask(cr * 256, g0, n=HANOVER_MASKN, fields=fields)
    hb = _hanover_both
    if HANOVER_COMPILE:
        if _hanover_both_c is None:
            _hanover_both_c = torch.compile(_hanover_both, dynamic=True)
        hb = _hanover_both_c
    obj2 = lambda: objective() + hb(cb, cr, wb, wr, float(lam), fields)
    opt = torch.optim.LBFGS([y, cb, cr], lr=lr, max_iter=iters, history_size=HIST,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
    def closure():
        opt.zero_grad(); l = obj2(); l.backward(); return l
    opt.step(closure)


def solve(C, cu, cv, init, lam_y=2e-3, eps_y=200., lam_cx=0.3, lam_cy=0.03, lam_hf=1.0, hf_mhz=5.6, iters=300, lr=60., method='adam', eps_c=0., noise_w=None,
          hanover=0., hanover_iters=30, hanover_g0=None, dmask=None):
    """One frame (C: (H, W)) or a batch (C: (B, H, W)); init (3, H, W) or (B, 3, H, W).
    dmask: optional per-pixel data weight shaped like C (dropouts 0: priors fill them)."""
    Dm = _dm(dmask, C)
    # unknowns in 8-bit units (x/256); see objective() for the loss scale
    init = np.asarray(init)
    y = torch.tensor(init[..., 0, :, :] / 256, device=dev, requires_grad=True)
    cb = torch.tensor((init[..., 1, :, :] - 32768) / 256, device=dev, requires_grad=True)
    cr = torch.tensor((init[..., 2, :, :] - 32768) / 256, device=dev, requires_grad=True)
    L0 = [None]
    nrm = 1.0 / C[..., :, :].shape[-1] / C.shape[-2]; hm = hf_mask(hf_mhz)

    def objective():
        Y, Cb, Cr = y * 256, cb * 256, cr * 256
        r = (M.P['AY'] * Y + M.P['B'] + Cb * cu + Cr * cv - C) * Dm
        loss = (r ** 2).sum()
        if lam_y:
            gx = Y[..., :, 1:] - Y[..., :, :-1]; gy = Y[..., 1:, :] - Y[..., :-1, :]
            loss = loss + lam_y * eps_y * (torch.sqrt(gx ** 2 + eps_y ** 2).sum() + torch.sqrt(gy ** 2 + eps_y ** 2).sum())
        if lam_hf:
            loss = loss + lam_hf * ((torch.abs(torch.fft.rfft(Y, dim=-1)) ** 2) * hm).sum() / OW
        for c in (Cb, Cr):
            dx = c[..., :, 1:] - c[..., :, :-1]; dy = c[..., 1:, :] - c[..., :-1, :]
            if eps_c:
                # edge-preserving chroma: a sharp colour edge (thin saturated map
                # lines) costs little, so it is not pushed into luma as ripple
                loss = loss + eps_c * (lam_cx * torch.sqrt(dx ** 2 + eps_c ** 2).sum() + lam_cy * torch.sqrt(dy ** 2 + eps_c ** 2).sum())
            else:
                loss = loss + lam_cx * (dx ** 2).sum() + lam_cy * (dy ** 2).sum()
        # sum of squared residuals in 8-bit units: per-pixel gradients ~1,
        # which L-BFGS's first step and its convergence tests assume
        return loss / 65536.0

    global _frame_loss_c, _frame_loss_wc
    if method == 'lbfgs' and eps_c and not lam_hf and noise_w is not None:
        if _frame_loss_wc is None:
            _frame_loss_wc = torch.compile(_frame_loss_w, dynamic=True)
        wf = torch.tensor(noise_w, device=dev)
        args = (C, cu, cv, Dm, wf, float(M.P['AY']), float(M.P['B']), float(lam_y), float(eps_y),
                float(lam_cx), float(lam_cy), float(eps_c))
        objective = lambda: _frame_loss_wc(y, cb, cr, *args)
    elif method == 'lbfgs' and eps_c and not lam_hf:
        # fast path: compiled fused objective (the tuned production settings)
        if _frame_loss_c is None:
            _frame_loss_c = torch.compile(_frame_loss, dynamic=True)
        args = (C, cu, cv, Dm, float(M.P['AY']), float(M.P['B']), float(lam_y), float(eps_y),
                float(lam_cx), float(lam_cy), float(eps_c))
        objective = lambda: _frame_loss_c(y, cb, cr, *args)
    if method == 'lbfgs':
        opt = torch.optim.LBFGS([y, cb, cr], lr=lr, max_iter=iters, history_size=HIST,
                                line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
        def closure():
            opt.zero_grad(); l = objective(); l.backward(); NEVAL[0] += 1; return l
        opt.step(closure)
        if hanover:
            del opt                      # free the stage-1 L-BFGS history before stage 2
            _hanover_stage(y, cb, cr, objective, hanover, hanover_iters, hanover_g0, lr)
    else:
        opt = torch.optim.Adam([y, cb, cr], lr=lr / 256)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for it in range(iters):
            opt.zero_grad(); objective().backward(); opt.step(); sched.step()
    return torch.stack([y * 256, cb * 256 + 32768, cr * 256 + 32768], dim=-3).detach().cpu().numpy()


def read_y4m(p):
    d = open(p, 'rb').read(); hdr = d[:d.index(b'\n') + 1]; n = OW * OH
    return hdr, [np.array([np.frombuffer(f[i * 2 * n:(i + 1) * 2 * n], '<u2').reshape(OH, OW) for i in range(3)]).astype(np.float32)
                 for f in d.split(b'FRAME\n')[1:]]


def field_setup(tbc, fields):
    """Per-field composite and carriers for a list of TBC field indices."""
    C, CU, CV = [], [], []
    for fld in fields:
        b0, sp = M.field_burst(tbc[fld])
        cu, cv = M.carriers(b0, sp, nlines=OH // 2, width=OW, x0=X0, line0=LINE0)
        C.append(tbc[fld, LINE0:LINE0 + OH // 2, X0:X0 + OW]); CU.append(cu); CV.append(cv)
    t = lambda x: torch.tensor(np.array(x, np.float32), device=dev)
    return t(C), t(CU), t(CV)


def _charb(d, eps):
    return eps * torch.sqrt(d ** 2 + eps ** 2).sum()


def solve_fields(C, cu, cv, init, parity, lam_y=0.02, eps_y=2000., lam_cx=0.3, lam_cy=0.1, eps_c=300.,
                 lam_t=0.05, eps_t=1000., lam_tc=0.3, lam_x=0.02, iters=60, lr=1e4, flow=None, noise_w=None,
                 lam_xm=0.0, lam_xmc=0.0, hp_pow=0.0, hp_pow1=0.0, lam_t4=0.0,
                 hanover=0., hanover_iters=30, hanover_g0=None, dmask=None):
    """Spatio-temporal solve of consecutive FIELDS (video): C (F, 288, W).

    Data term per field (exact model).  Spatial priors within each field
    (its lines are two frame lines apart, so vertical priors are halved).
    Temporal priors, edge-preserving so real motion is not averaged away:
    each field against the same-parity field one frame earlier (lam_t, lam_tc
    for chroma), and against the vertically interpolated opposite field just
    before it (lam_x).  noise_w: optional per-frequency data-term weights
    (noise_weights), as in the stills and film solvers.

    lam_xm / lam_xmc (luma / chroma; needs flow): motion-compensated
    opposite-field term, each field against the field just before it (other
    parity, half a frame earlier) warped by half the lag-2 flow plus the
    half-line parity offset.  On the video bench (2026-10-14) it is worth
    +0.4 to +1.7 dB luma, most on vertical motion.
    hp_pow > 0 (needs flow): the lag-2 term compares vertically low-passed
    differences ([1,2,1]/4), plus the vertical-HF part of the luma difference
    weighted by |cos(pi * frac(dy))| ** hp_pow: a field shifted by a fraction
    of a field line cannot be resampled from its aliased 288 lines, so its
    vertical detail is only trusted where the motion is close to whole field
    lines (static, horizontal pans, 2 frame lines/frame).
    hp_pow1 > 0: the same split for the luma of the lam_xm term, weighted by
    the fractional part of its vertical shift (half a line for static
    content, where the lag-2 term already carries the vertical detail).
    lam_t4 (needs flow and hp_pow): a second same-parity term two frames back
    (index f-4, flow = sum of two lag-2 flows, constant velocity), with the
    same low-pass / hp_pow split; chroma weight lam_tc * lam_t4 / lam_t.  A
    1-line/frame vertical pan is a whole field line at lag 4."""
    init = np.asarray(init)
    y = torch.tensor(init[:, 0] / 256, device=dev, requires_grad=True)
    cb = torch.tensor((init[:, 1] - 32768) / 256, device=dev, requires_grad=True)
    cr = torch.tensor((init[:, 2] - 32768) / 256, device=dev, requires_grad=True)
    par = list(parity)
    Dm = _dm(dmask, C)
    FL = torch.tensor(flow[2:], device=dev) if flow is not None else None
    wf = torch.tensor(noise_w, device=dev) if noise_w is not None else None
    FL1 = HP = HP1 = HP4 = G4 = None
    if flow is not None and (lam_xm or lam_xmc):
        h = np.array(flow, np.float32)
        h[:-1] = 0.5 * (h[:-1] + h[1:]); h[:2] = h[2]          # flow over the interval f-1 .. f
        h = 0.5 * h
        h[..., 1] += np.where(np.array(par) == 1, -0.5, 0.5)[:, None, None]   # half-line parity offset
        FL1 = torch.tensor(h[1:], device=dev)
        HP1 = torch.tensor(np.abs(np.cos(np.pi * (h[1:, ..., 1] - np.round(h[1:, ..., 1])))) ** hp_pow1, device=dev) if hp_pow1 else None
    if flow is not None and hp_pow:
        dy = np.asarray(flow[2:, ..., 1], np.float32)
        HP = torch.tensor(np.abs(np.cos(np.pi * (dy - np.round(dy)))) ** hp_pow, device=dev)
    if flow is not None and hp_pow and lam_t4 and len(init) > 4:
        f4 = np.array(flow, np.float32); f4[2:] = f4[2:] + np.asarray(flow[:-2], np.float32)
        dy = f4[4:, ..., 1]
        HP4 = torch.tensor(np.abs(np.cos(np.pi * (dy - np.round(dy)))) ** hp_pow, device=dev)
        FL4 = torch.tensor(f4[4:], device=dev)
    lam_tc4 = lam_tc * lam_t4 / lam_t if lam_t else lam_tc
    vl = lambda d: 0.25 * (d[:, :-2] + 2 * d[:, 1:-1] + d[:, 2:])
    import torch.nn.functional as Fn
    def grid(fl):            # sampling grid for p - fl, built once (not per evaluation)
        h_, w_ = fl.shape[1:3]
        yy, xx = torch.meshgrid(torch.arange(h_, device=dev, dtype=torch.float32),
                                torch.arange(w_, device=dev, dtype=torch.float32), indexing='ij')
        return torch.stack([(xx[None] - fl[..., 0]) / (w_ - 1) * 2 - 1, (yy[None] - fl[..., 1]) / (h_ - 1) * 2 - 1], -1)
    G2 = grid(FL) if FL is not None else None
    G1 = grid(FL1) if FL1 is not None else None
    G4 = grid(FL4) if HP4 is not None else None
    # Y, Cb, Cr warped together (one grid_sample with 3 channels)
    wp3 = lambda Z, G: Fn.grid_sample(Z, G, mode='bicubic', padding_mode='border', align_corners=True).unbind(1)

    def interp_to(prev, p_from, p_to):
        # opposite field resampled onto this field's lines (half-line offset)
        if p_from == 0 and p_to == 1:      # rows 2i+1 between 2i and 2i+2
            return 0.5 * (prev + torch.cat([prev[1:], prev[-1:]], 0))
        return 0.5 * (prev + torch.cat([prev[:1], prev[:-1]], 0))   # rows 2i between 2i-1 and 2i+1

    def objective():
        Y, Cb, Cr = y * 256, cb * 256, cr * 256
        r = (M.P['AY'] * Y + M.P['B'] + Cb * cu + Cr * cv - C) * Dm
        if wf is None:
            loss = (r ** 2).sum()
        else:
            R = torch.fft.rfft(r, dim=-1)
            loss = ((R.real ** 2 + R.imag ** 2) * wf).sum() * (2.0 / r.shape[-1])
        loss = loss + lam_y * (_charb(Y[..., :, 1:] - Y[..., :, :-1], eps_y) + 0.5 * _charb(Y[..., 1:, :] - Y[..., :-1, :], eps_y))
        for c in (Cb, Cr):
            loss = loss + lam_cx * _charb(c[..., :, 1:] - c[..., :, :-1], eps_c) + 0.5 * lam_cy * _charb(c[..., 1:, :] - c[..., :-1, :], eps_c)
        if lam_t and Y.shape[0] > 2:
            # previous same-parity field, motion-compensated when a flow is given
            if G2 is not None:
                PY, PCb, PCr = wp3(torch.stack([Y[:-2], Cb[:-2], Cr[:-2]], 1), G2)
            else:
                PY, PCb, PCr = Y[:-2], Cb[:-2], Cr[:-2]
            if HP is None:
                loss = loss + lam_t * _charb(Y[2:] - PY, eps_t)
                for c, pc in ((Cb, PCb), (Cr, PCr)):
                    loss = loss + lam_tc * _charb(c[2:] - pc, eps_c)
            else:
                d = Y[2:] - PY; dl = vl(d)
                loss = loss + lam_t * (_charb(dl, eps_t) + eps_t * (HP[:, 1:-1] * torch.sqrt((d[:, 1:-1] - dl) ** 2 + eps_t ** 2)).sum())
                for c, pc in ((Cb, PCb), (Cr, PCr)):
                    loss = loss + lam_tc * _charb(vl(c[2:] - pc), eps_c)
        if G4 is not None:
            PY, PCb, PCr = wp3(torch.stack([Y[:-4], Cb[:-4], Cr[:-4]], 1), G4)
            d = Y[4:] - PY; dl = vl(d)
            loss = loss + lam_t4 * (_charb(dl, eps_t) + eps_t * (HP4[:, 1:-1] * torch.sqrt((d[:, 1:-1] - dl) ** 2 + eps_t ** 2)).sum())
            for c, pc in ((Cb, PCb), (Cr, PCr)):
                loss = loss + lam_tc4 * _charb(vl(c[4:] - pc), eps_c)
        if G1 is not None and Y.shape[0] > 1:
            PY, PCb, PCr = wp3(torch.stack([Y[:-1], Cb[:-1], Cr[:-1]], 1), G1)
            if lam_xm and HP1 is None:
                loss = loss + lam_xm * _charb(Y[1:] - PY, eps_t)
            elif lam_xm:
                d = Y[1:] - PY; dl = vl(d)
                loss = loss + lam_xm * (_charb(dl, eps_t) + eps_t * (HP1[:, 1:-1] * torch.sqrt((d[:, 1:-1] - dl) ** 2 + eps_t ** 2)).sum())
            if lam_xmc:
                for c, pc in ((Cb, PCb), (Cr, PCr)):
                    loss = loss + lam_xmc * _charb(c[1:] - pc, eps_c)
        if lam_x and Y.shape[0] > 1:
            for f in range(1, Y.shape[0]):
                loss = loss + lam_x * _charb(Y[f] - interp_to(Y[f - 1], par[f - 1], par[f]), eps_t)
        return loss / 65536.0

    global _fields_loss_c
    if wf is not None and HP is not None and HP1 is not None and lam_t and lam_xm and lam_xmc and not lam_x and len(init) > 2:
        # fast path (the recommended settings): the same objective, compiled
        # (1.4x faster than eager on the RTX 3060, and ~0.4 GB less memory)
        if _fields_loss_c is None:
            _fields_loss_c = torch.compile(_fields_loss, dynamic=True)
        consts = tuple(float(v) for v in (M.P['AY'], M.P['B'], lam_y, eps_y, lam_cx, lam_cy, eps_c,
                                          lam_t, eps_t, lam_tc, lam_xm, lam_xmc, lam_t4 if G4 is not None else 0.0, lam_tc4))
        g4, h4 = (G4, HP4) if G4 is not None else (G2[:1], HP[:1])     # dummies when lam_t4 is off
        objective = lambda: _fields_loss_c(y, cb, cr, C, cu, cv, Dm, wf, G2, G1, HP, HP1, g4, h4, *consts)
    opt = torch.optim.LBFGS([y, cb, cr], lr=lr, max_iter=iters, history_size=HIST,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
    def closure():
        opt.zero_grad(); l = objective(); l.backward(); return l
    opt.step(closure)
    if hanover:
        del opt                          # free the stage-1 L-BFGS history before stage 2
        _hanover_stage(y, cb, cr, objective, hanover, hanover_iters, hanover_g0, lr, fields=True)
    return torch.stack([y * 256, cb * 256 + 32768, cr * 256 + 32768], dim=1).detach().cpu().numpy()


def _vl(d):
    return 0.25 * (d[:, :-2] + 2 * d[:, 1:-1] + d[:, 2:])


def _fields_loss(y, cb, cr, C, cu, cv, Dm, wf, G2, G1, HP, HP1, G4, HP4, AY, B0, lam_y, eps_y, lam_cx, lam_cy, eps_c,
                 lam_t, eps_t, lam_tc, lam_xm, lam_xmc, lam_t4, lam_tc4):
    """solve_fields' objective with flow, noise_w, lam_xm/lam_xmc and both
    hp_pow splits on, as one function of tensors for torch.compile."""
    import torch.nn.functional as Fn
    wp3 = lambda Z, G: Fn.grid_sample(Z, G, mode='bicubic', padding_mode='border', align_corners=True)
    Y, Cb, Cr = y * 256, cb * 256, cr * 256
    r = (AY * Y + B0 + Cb * cu + Cr * cv - C) * Dm
    R = torch.fft.rfft(r, dim=-1)
    loss = ((R.real ** 2 + R.imag ** 2) * wf).sum() * (2.0 / r.shape[-1])
    loss = loss + lam_y * (_charb(Y[..., :, 1:] - Y[..., :, :-1], eps_y) + 0.5 * _charb(Y[..., 1:, :] - Y[..., :-1, :], eps_y))
    for c in (Cb, Cr):
        loss = loss + lam_cx * _charb(c[..., :, 1:] - c[..., :, :-1], eps_c) + 0.5 * lam_cy * _charb(c[..., 1:, :] - c[..., :-1, :], eps_c)
    W = wp3(torch.stack([Y[:-2], Cb[:-2], Cr[:-2]], 1), G2)
    d = Y[2:] - W[:, 0]; dl = _vl(d)
    loss = loss + lam_t * (_charb(dl, eps_t) + eps_t * (HP[:, 1:-1] * torch.sqrt((d[:, 1:-1] - dl) ** 2 + eps_t ** 2)).sum())
    loss = loss + lam_tc * (_charb(_vl(Cb[2:] - W[:, 1]), eps_c) + _charb(_vl(Cr[2:] - W[:, 2]), eps_c))
    W = wp3(torch.stack([Y[:-1], Cb[:-1], Cr[:-1]], 1), G1)
    d = Y[1:] - W[:, 0]; dl = _vl(d)
    loss = loss + lam_xm * (_charb(dl, eps_t) + eps_t * (HP1[:, 1:-1] * torch.sqrt((d[:, 1:-1] - dl) ** 2 + eps_t ** 2)).sum())
    loss = loss + lam_xmc * (_charb(Cb[1:] - W[:, 1], eps_c) + _charb(Cr[1:] - W[:, 2], eps_c))
    if lam_t4:
        W = wp3(torch.stack([Y[:-4], Cb[:-4], Cr[:-4]], 1), G4)
        d = Y[4:] - W[:, 0]; dl = _vl(d)
        loss = loss + lam_t4 * (_charb(dl, eps_t) + eps_t * (HP4[:, 1:-1] * torch.sqrt((d[:, 1:-1] - dl) ** 2 + eps_t ** 2)).sum())
        loss = loss + lam_tc4 * (_charb(_vl(Cb[4:] - W[:, 1]), eps_c) + _charb(_vl(Cr[4:] - W[:, 2]), eps_c))
    return loss / 65536.0


_fields_loss_c = None


def estimate_flow(Yf, bx=64, by=32, sx=32, sy=16, lag=2):
    """Motion of each field relative to the same-parity field one frame
    earlier (index f-2), from block phase correlation on luma.
    Yf: (F, h, w) numpy.  Returns (F, h, w, 2) displacement (dx, dy) in
    samples / field lines such that Yf[f](p) ~ Yf[f-2](p - d)."""
    F, h, w = Yf.shape
    out = np.zeros((F, h, w, 2), np.float32)
    win = np.outer(np.hanning(by), np.hanning(bx))
    ys = list(range(0, h - by + 1, sy)); xs = list(range(0, w - bx + 1, sx))
    for f in range(lag, F):
        D = np.zeros((len(ys), len(xs), 2), np.float32)
        for i, y0 in enumerate(ys):
            for j, x0 in enumerate(xs):
                a = Yf[f - lag, y0:y0 + by, x0:x0 + bx]; b = Yf[f, y0:y0 + by, x0:x0 + bx]
                A = np.fft.fft2((a - a.mean()) * win); B = np.fft.fft2((b - b.mean()) * win)
                X = B * np.conj(A); X /= np.abs(X) + 1e-3 * np.abs(X).max() + 1e-9
                c = np.real(np.fft.ifft2(X))
                py, px = np.unravel_index(np.argmax(c), c.shape)
                def sub(m, z, p):
                    d = m - 2 * z + p
                    return 0.0 if d == 0 else 0.5 * (m - p) / d
                dy = py + sub(c[py - 1, px], c[py, px], c[(py + 1) % by, px])
                dx = px + sub(c[py, px - 1], c[py, px], c[py, (px + 1) % bx])
                if dy > by / 2: dy -= by
                if dx > bx / 2: dx -= bx
                D[i, j] = dx, dy
        from scipy.ndimage import median_filter, zoom
        D = np.stack([median_filter(D[..., k], 3) for k in range(2)], -1)
        full = np.stack([zoom(D[..., k], (h / len(ys), w / len(xs)), order=1) for k in range(2)], -1)
        out[f] = full[:h, :w]
    return out


def _warp(img, flow):
    """Sample img (N, h, w) at p - flow (flow (N, h, w, 2) torch, in pixels)."""
    import torch.nn.functional as Fn
    N, h, w = img.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=img.device, dtype=img.dtype),
                            torch.arange(w, device=img.device, dtype=img.dtype), indexing='ij')
    gx = (xx[None] - flow[..., 0]) / (w - 1) * 2 - 1
    gy = (yy[None] - flow[..., 1]) / (h - 1) * 2 - 1
    return Fn.grid_sample(img[:, None], torch.stack([gx, gy], -1), mode='bicubic',
                          padding_mode='border', align_corners=True)[:, 0]


def _film_loss(y, cb, cr, C, cu, cv, Dm, wf, grid, tm, AY, B0, lam_y, eps_y, lam_cx, lam_cy, eps_c, lam_t, eps_t, lam_tc):
    """solve_film's objective as one function of tensors so torch.compile can
    fuse it; `grid` is the motion-compensating sampling grid, built once; tm (N, H, W) the temporal prior's
    per-pixel weight (film_tmask)."""
    import torch.nn.functional as Fn
    l = _frame_loss_w(y, cb, cr, C, cu, cv, Dm, wf, AY, B0, lam_y, eps_y, lam_cx, lam_cy, eps_c) * 65536.0
    Y, Cb, Cr = y * 256, cb * 256, cr * 256
    wp = lambda x: Fn.grid_sample(x[:-1, None], grid, mode='bicubic', padding_mode='border', align_corners=True)[:, 0]
    ch = lambda d, e: e * (tm * torch.sqrt(d ** 2 + e ** 2)).sum()
    l = l + lam_t * ch(Y[1:] - wp(Y), eps_t)
    l = l + lam_tc * (ch(Cb[1:] - wp(Cb), eps_c) + ch(Cr[1:] - wp(Cr), eps_c))
    return l / 65536.0


FILM_TM_LO = float(os.environ.get('PALDEC_FILM_TM_LO', 3000.))   # 16-bit; 0 = off (v6_0: temporal prior everywhere)
FILM_TM_HI = float(os.environ.get('PALDEC_FILM_TM_HI', 8000.))
FILM_TM_MIN = float(os.environ.get('PALDEC_FILM_TM_MIN', 0.05))


def film_tmask(init, grid):
    """2026-10-28 (herring integration): where a frame's init differs from the motion-compensated previous init
    by far more than noise (a cursor / highlight / text change inside a film or held run: synthetic 3-state test
    ghost 24 8-bit levels), the temporal prior is weakened there (x FILM_TM_MIN above FILM_TM_HI), so the change
    is not averaged with its neighbour.  -> (N, H, W) torch"""
    import torch.nn.functional as Fn
    Yi = torch.as_tensor(np.asarray(init)[:, 0], dtype=torch.float32, device=dev)
    if FILM_TM_LO <= 0:
        return torch.ones_like(Yi[1:])
    d = (Yi[1:] - Fn.grid_sample(Yi[:-1, None], grid, mode='bilinear', padding_mode='border', align_corners=True)[:, 0]).abs()
    k = torch.ones((1, 1, 5, 9), device=dev) / 45.0
    d = Fn.conv2d(d[:, None], k, padding=(2, 4))[:, 0]
    a = ((d - FILM_TM_LO) / (FILM_TM_HI - FILM_TM_LO)).clamp(0, 1)
    a = Fn.max_pool2d(a[:, None], (5, 9), stride=1, padding=(2, 4))[:, 0]
    return 1 - (1 - FILM_TM_MIN) * a


_film_loss_c = None


def solve_film(C, cu, cv, init, flow, noise_w=None, lam_y=0.04, eps_y=1200., lam_cx=0.3, lam_cy=0.1,
               eps_c=300., lam_t=0.2, eps_t=300., lam_tc=0.3, iters=60, lr=1e4,
               hanover=0., hanover_iters=30, hanover_g0=None, dmask=None):
    """Consecutive FRAMES of film / held stills (both fields one instant):
    the stills objective per frame (noise-weighted data term, spatial priors
    on the woven frame) plus edge-preserving temporal priors tying each frame
    to the previous one, motion-compensated by `flow` (F, H, W, 2), estimated
    at full frame resolution (estimate_flow(..., lag=1))."""
    init = np.asarray(init)
    y = torch.tensor(init[:, 0] / 256, device=dev, requires_grad=True)
    cb = torch.tensor((init[:, 1] - 32768) / 256, device=dev, requires_grad=True)
    cr = torch.tensor((init[:, 2] - 32768) / 256, device=dev, requires_grad=True)
    wf = torch.tensor(noise_w if noise_w is not None else np.ones(OW // 2 + 1, np.float32), device=dev)
    FL = torch.tensor(flow[1:], device=dev)
    Dm = _dm(dmask, C)
    AY, B0 = float(M.P['AY']), float(M.P['B'])
    global _film_loss_c
    if _film_loss_c is None:
        _film_loss_c = torch.compile(_film_loss, dynamic=True)
    N, h, w = y.shape[0] - 1, y.shape[1], y.shape[2]
    yy, xx = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing='ij')
    grid = torch.stack([(xx[None] - FL[..., 0]) / (w - 1) * 2 - 1,
                        (yy[None] - FL[..., 1]) / (h - 1) * 2 - 1], -1)
    consts = (AY, B0, float(lam_y), float(eps_y), float(lam_cx), float(lam_cy), float(eps_c),
              float(lam_t), float(eps_t), float(lam_tc))
    tm = film_tmask(init, grid)
    objective = lambda: _film_loss_c(y, cb, cr, C, cu, cv, Dm, wf, grid, tm, *consts)

    opt = torch.optim.LBFGS([y, cb, cr], lr=lr, max_iter=iters, history_size=HIST,
                            line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
    def closure():
        opt.zero_grad(); l = objective(); l.backward(); return l
    opt.step(closure)
    if hanover:
        del opt                          # free the stage-1 L-BFGS history before stage 2
        _hanover_stage(y, cb, cr, objective, hanover, hanover_iters, hanover_g0, lr)
    return torch.stack([y * 256, cb * 256 + 32768, cr * 256 + 32768], dim=1).detach().cpu().numpy()


def estimate_flow_gpu(Yf, bx=64, by=32, sx=32, sy=16, lag=1):
    """GPU version of estimate_flow: all blocks of all frame pairs in one
    batched phase correlation (same blocks, window, sub-pixel fit and 3x3
    median smoothing).  Yf: (F, h, w) numpy or tensor.  Returns numpy flow."""
    import torch.nn.functional as Fn
    Y = torch.as_tensor(np.asarray(Yf, np.float32), device=dev)
    F_, h, w = Y.shape
    out = torch.zeros((F_, h, w, 2), device=dev)
    if F_ <= lag:
        return out.cpu().numpy()
    win = torch.outer(torch.hann_window(by, periodic=False, device=dev), torch.hann_window(bx, periodic=False, device=dev))
    def blocks(x):                    # (N, h, w) -> (N, ny, nx, by, bx)
        b = x.unfold(1, by, sy).unfold(2, bx, sx)
        return b - b.mean((-1, -2), keepdim=True)
    A = torch.fft.fft2(blocks(Y[:-lag]) * win); B = torch.fft.fft2(blocks(Y[lag:]) * win)
    X = B * A.conj()
    X = X / (X.abs() + 1e-3 * X.abs().amax((-1, -2), keepdim=True) + 1e-9)
    c = torch.fft.ifft2(X).real                                  # (N, ny, nx, by, bx)
    N, ny, nx = c.shape[:3]
    flat = c.reshape(N, ny, nx, -1).argmax(-1)
    py, px = flat // bx, flat % bx
    g = lambda dy, dx: torch.gather(c.reshape(N, ny, nx, -1), -1, (((py + dy) % by) * bx + (px + dx) % bx)[..., None])[..., 0]
    def sub(m, z, p):
        d = m - 2 * z + p
        return torch.where(d == 0, torch.zeros_like(d), 0.5 * (m - p) / torch.where(d == 0, torch.ones_like(d), d))
    z = g(0, 0)
    dy = py.float() + sub(g(-1, 0), z, g(1, 0)); dx = px.float() + sub(g(0, -1), z, g(0, 1))
    dy = torch.where(dy > by / 2, dy - by, dy); dx = torch.where(dx > bx / 2, dx - bx, dx)
    D = torch.stack([dx, dy], 1)                                 # (N, 2, ny, nx)
    # 3x3 median, then bilinear up to full resolution
    p = Fn.pad(D, (1, 1, 1, 1), mode='replicate').unfold(2, 3, 1).unfold(3, 3, 1)
    D = p.reshape(*p.shape[:4], 9).median(-1).values
    full = Fn.interpolate(D, size=(h, w), mode='bilinear', align_corners=True)
    out[lag:] = full.permute(0, 2, 3, 1)
    return out.cpu().numpy()


# ---- held stills: combine every repeat of a static run (2026-10-14) ----
# A held still is a run of separate disc frames carrying the identical picture, each with
# independent disc noise.  After the per-frame film solve (the candidate path):
#   1. static gate (held_static): no pan/drift between the frames' solutions
#   2. one shared latent is solved from every repeat's own (carrier-phase-correct) data term,
#      with per-frame corrections for the real discs' frame-to-frame chroma crosstalk
#   3. per-frame, per-16x16-block outliers: composite-domain residual against that latent
#      (held_outliers) plus a model-free decoded-domain test (held_decoded_check); outlier
#      blocks are dilated by one block and feathered (held_alpha)
#   4. output, PALDEC_HELD_MODE=avg (default): the inlier-weighted MEAN of the per-frame
#      solutions where the frame agrees, its own solution elsewhere.  =joint: the shared latent
#      instead (best on the synthetic bench, noisier than avg on real South clips).
#   PALDEC_HELD=1|0         on/off (0 = the combined candidate exactly)
#   PALDEC_HELD_MODE=avg|joint
#   PALDEC_HELD_MIN=3       minimum run length
#   PALDEC_HELD_T=3         outlier threshold: block energy of the frame's band-limited residual (minus
#                           the run median) vs the global median block energy
#   PALDEC_HELD_BLOCK=16    outlier block size (pixels)
#   PALDEC_HELD_ITERS=40    L-BFGS iterations for the shared latent (init: inlier mean)
#   PALDEC_HELD_PS=4        spatial prior scale for the shared latent (x the stills priors x sqrt(max(1, m/8)))
#   PALDEC_HELD_PSC=1       extra scale on its chroma priors
#   PALDEC_HELD_T1=10       first outlier pass threshold multiplier (gross changes only)
#   PALDEC_HELD_DGRID=32    per-frame coarse mismatch-correction grid (px), 0 = off
#   PALDEC_HELD_DMAX=0 (off) / PALDEC_HELD_DMEAN=6   optional decoded-domain outlier test: blurred |frame - inlier
#                           mean| (minus the frame's global drift), block mean / block max, 8-bit levels
#   PALDEC_HELD_FLOW=0.1    static gate: median over frame pairs of |median flow vector| (px), lags 1 and m/2
HELD = os.environ.get('PALDEC_HELD', '1') == '1'
HELD_MIN = int(os.environ.get('PALDEC_HELD_MIN', 3))
HELD_LOCN = os.environ.get('PALDEC_HELD_LOCN', '1') == '1'   # v6: outlier threshold also relative to the block's own median
# v6: 5 with HELD_LOCN (a 16x16 block of the blurred residual has few degrees of freedom: 3x its own median is
# reached by noise alone in several % of blocks; bars 3092 green 9.5 -> 4.3, a cursor / highlight is 10-100x)
HELD_T = float(os.environ.get('PALDEC_HELD_T', 5.0 if HELD_LOCN else 3.0))
HELD_LOCK = float(os.environ.get('PALDEC_HELD_LOCK', 10.0))   # v6: cap of the local term, x the global median
HELD_Q = os.environ.get('PALDEC_HELD_Q', '1') == '1'           # v6 10-27: lower quartile + cap (0: v6_0's own median, ghosting)
HELD_BLOCK = int(os.environ.get('PALDEC_HELD_BLOCK', 16))
HELD_ITERS = int(os.environ.get('PALDEC_HELD_ITERS', 40))
HELD_PS = float(os.environ.get('PALDEC_HELD_PS', 4.0))
HELD_PSC = float(os.environ.get('PALDEC_HELD_PSC', 1.0))     # extra chroma prior scale
HELD_FLOW = float(os.environ.get('PALDEC_HELD_FLOW', 0.1))
HELD_DRIFT = float(os.environ.get('PALDEC_HELD_DRIFT', 0.5))     # px over the run; 0 = off (v6_0)
HELD_MOVK = float(os.environ.get('PALDEC_HELD_MOVK', 6.0))
HELD_MOVFRAC = float(os.environ.get('PALDEC_HELD_MOVFRAC', 0.02))   # 0 = off (v6_0)
HELD_APERTURE = float(os.environ.get('PALDEC_HELD_APERTURE', 0.1))   # 0: v6_0 (both flow components always tested)
HELD_T1 = float(os.environ.get('PALDEC_HELD_T1', 10))      # first (gross) outlier pass: T x T1, before the mismatch fit
HELD_DGRID = int(os.environ.get('PALDEC_HELD_DGRID', 32))  # per-frame mismatch correction grid (px; 0 = off)
HELD_LAMD = float(os.environ.get('PALDEC_HELD_LAMD', 1e-4))  # its (weak) size penalty
HELD_MATRIX = os.environ.get('PALDEC_HELD_MATRIX', '1') == '1'   # per-frame/row-class 2x2 chroma crosstalk matrices
HELD_MODE = os.environ.get('PALDEC_HELD_MODE', 'avg')   # avg: robust mean of the per-frame solutions; joint: the shared latent
HELD_DMAX = float(os.environ.get('PALDEC_HELD_DMAX', 0.0))    # decoded-domain outlier test (8-bit): block max; 0 = test off (default: it rejects text/graphics edges, whose per-frame decode differs by subcarrier phase)
HELD_DMEAN = float(os.environ.get('PALDEC_HELD_DMEAN', 6.0))  # ... and block mean
HELD_RETEST_D = os.environ.get('PALDEC_HELD_RETEST_D', '0') == '1'
HELD_DOW = os.environ.get('PALDEC_HELD_DOW', '1') == '1'     # v6: avg mode weights frames by their dropout mask
HELD_DEBUG = {}
HELD_KEEP = False                  # benches: keep the last weights in HELD_DEBUG['alpha']


def _gblur(x, sx=2.0, sy=1.0):
    """separable Gaussian blur of (N, 1, H, W)"""
    import torch.nn.functional as Fn
    def k(s):
        r = int(3 * s + 0.5); t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        g = torch.exp(-0.5 * (t / s) ** 2); return g / g.sum()
    kx = k(sx); ky = k(sy)
    x = Fn.conv2d(Fn.pad(x, (len(kx) // 2,) * 2 + (0, 0), mode='replicate'), kx[None, None, None])
    return Fn.conv2d(Fn.pad(x, (0, 0) + (len(ky) // 2,) * 2, mode='replicate'), ky[None, None, :, None])


def held_outliers(Z, C, cu, cv, block=None, T=None, Dm=1.0):
    """Per-frame outlier blocks of a held run, tested in the COMPOSITE domain against a
    consensus picture Z (3, H, W) torch (16-bit units).  Each frame's residual against Z
    re-encoded with that frame's own carriers is split into a luma band (Gaussian
    low-pass, below ~1.5 MHz, where disc noise is low) and chroma (demodulated with the
    frame's own carriers, low-passed).  Where all frames show the same picture these
    band-limited residuals are disc noise plus Z's own error, which is the SAME for every
    frame; subtracting the per-pixel median over frames cancels it, leaving noise (the same
    level everywhere) plus any frame-specific content.  A block is an outlier when either
    band's energy exceeds T x the global median block energy (v6: x max(that, min(the block's own lower
    quartile over the frames, HELD_LOCK x that)); PALDEC_HELD_Q=0: its median, no cap).  -> bool (m, 1, nby, nbx)"""
    import torch.nn.functional as Fn
    block = block or HELD_BLOCK; T = T or HELD_T
    with torch.no_grad():
        r = (M.P['AY'] * Z[0] + M.P['B'] + (Z[1] - 32768) * cu + (Z[2] - 32768) * cv - C) * Dm     # (m, H, W)
        bands = [[_gblur(r[:, None])], [_gblur((r * cu)[:, None]), _gblur((r * cv)[:, None])]]
        del r
        m, _, H, W = bands[0][0].shape
        # v6: each frame's composite carries chroma gain / phase errors and sub-sample timing jitter the model leaves
        # in the residual, so a saturated or edge block's residual is larger in EVERY frame; against one global
        # threshold such blocks were outliers in a fraction of the frames, and with the one-block dilation they
        # kept their own (unaveraged) solution - grain in saturated flat colour (South bars 3092, 10-bit luma rms
        # cyan / green 16 / 12 vs white 1.3).  A block is now tested against T x max(n0, min(q, HELD_LOCK x n0)):
        # n0 the global median, q the block's lower quartile over the run's frames.  The quartile (not the median)
        # still flags a change that appears mid-run and stays (half the frames or more: text, a highlight), and
        # the cap bounds the local term if it covers nearly all frames (PALDEC_HELD_LOCN=0: v5)
        kq = max(1, m // 4)
        def test(center):
            out = None
            for fs in bands:
                e = sum(Fn.avg_pool2d((f - center(f)) ** 2, block) for f in fs)
                n0 = e.median()
                if HELD_LOCN and HELD_Q:
                    q = e.kthvalue(kq, dim=0, keepdim=True).values
                    o = e > T * torch.clamp(q, min=float(n0), max=float(HELD_LOCK * n0))
                else:
                    o = e > T * (torch.maximum(e.median(0, keepdim=True).values, n0) if HELD_LOCN else n0)
                out = o if out is None else out | o
            return out
        out = test(lambda f: f.median(0, keepdim=True).values)
        # second pass: the median of a run with a minority of changed frames is biased
        # towards them; centre on the mean of the frames that passed instead
        w = Fn.interpolate((~out).float(), size=(H // block * block, W // block * block), mode='nearest')
        w = Fn.pad(w, (0, W - w.shape[-1], 0, H - w.shape[-2]), mode='replicate')
        n = w.sum(0, keepdim=True)
        def cmean(f):
            return torch.where(n > 0, (f * w).sum(0, keepdim=True) / n.clamp(min=1), f.median(0, keepdim=True).values)
        out = test(cmean)
    return out


def held_decoded_check(Xt, out, block=None, rounds=2):
    """Second, model-free outlier test in the decoded domain: a block of frame i is also an
    outlier where its own solution differs from the inlier mean by more than HELD_DMAX
    (8-bit levels, Gaussian-blurred luma or chroma, minus the frame's fitted global drift): block
    mean > HELD_DMEAN or block max > HELD_DMAX.  A moving cursor or a highlight is 10-100."""
    import torch.nn.functional as Fn
    if not HELD_DMAX:
        return out
    block = block or HELD_BLOCK
    m, _, H, W = Xt.shape
    with torch.no_grad():
        for _ in range(rounds):
            w = Fn.interpolate((~out).float(), size=(H // block * block, W // block * block), mode='nearest')
            w = Fn.pad(w, (0, W - w.shape[-1], 0, H - w.shape[-2]), mode='replicate')        # (m, 1, H, W)
            mean = (Xt * w).sum(0) / w.sum(0).clamp(min=1)
            new = torch.zeros_like(out)
            # a frame's global level / gain / hue drift (real discs: chroma drifts several levels,
            # content-dependent, over a run) is not a local change: it is fitted per frame as a
            # linear function of the mean picture's blurred Y, Cb, Cr (+ const), removed from the
            # test, and averaged away in the output
            Bm = torch.cat([torch.ones((1, H * W), device=Xt.device)] +
                           [_gblur((mean[c:c + 1][None] - (32768 if c else 0)) / 256, 1.5, 1.5).reshape(1, -1) for c in range(3)])  # (4, P)
            BtB = Bm @ Bm.T
            for ch in range(3):
                g = _gblur((Xt[:, ch:ch + 1] - mean[ch]) / 256, 1.5, 1.5).reshape(m, -1)
                coef = torch.linalg.solve(BtB, Bm @ g.T)                    # (4, m)
                g = (g - (coef.T @ Bm)).abs().reshape(m, 1, H, W)
                new |= (Fn.avg_pool2d(g, block) > HELD_DMEAN) | (Fn.max_pool2d(g, block) > HELD_DMAX)
            del g, w, mean, Bm
            if not (new & ~out).any():
                break
            out = out | new
    HELD_DEBUG['dec_out'] = int(out.sum())
    return out


def held_alpha(out, H, W, block=None):
    """outlier blocks (m, 1, nby, nbx) bool -> per-frame pixel weights (m, H, W): 0 in the
    outlier blocks dilated by one block, bilinear feather to 1.  A frame with fewer than
    half its blocks inlying (motion) gets 0; a block needs >= 2 inlying frames."""
    import torch.nn.functional as Fn
    block = block or HELD_BLOCK
    inl = 1.0 - Fn.max_pool2d(out.float(), 3, 1, 1)
    frac = inl.mean((1, 2, 3))
    inl[frac < 0.5] = 0
    inl = inl * (inl.sum(0, keepdim=True) >= 2)
    HELD_DEBUG['frac'] = frac.cpu().numpy(); HELD_DEBUG['outblocks'] = int(out.sum())
    a = Fn.interpolate(inl, size=(H // block * block, W // block * block), mode='bilinear', align_corners=False)
    a = Fn.pad(a, (0, W - a.shape[-1], 0, H - a.shape[-2]), mode='replicate')
    if HELD_KEEP:
        HELD_DEBUG['alpha'] = a[:, 0].cpu().numpy()
    return a[:, 0]


def _held_mismatch(dc, dl, H, W):
    """per-frame coarse corrections -> (dY (m, H, W), dCb, dCr (m, H, W)), 8-bit units.
    dc (m, 8, gy, gx): chroma offsets per frame for each of the 4 frame-row classes (field
    parity x V-switch line parity), zero mean over frames and classes; dl (m, 4, gy, gx):
    luma offsets per frame and row class, zero mean over frames and classes.  Bilinear up to full resolution."""
    import torch.nn.functional as Fn
    m = dc.shape[0]
    dc = dc - dc.view(m, 4, 2, *dc.shape[-2:]).mean((0, 1)).repeat(4, 1, 1)[None]
    dl = dl - dl.mean((0, 1), keepdim=True)
    gy, gx = dc.shape[-2:]
    up = Fn.interpolate(torch.cat([dc.view(m, 4, 2, gy, gx), dl.view(m, 4, 1, gy, gx)], 2).view(m, 12, gy, gx), size=(H // 4, W), mode='bilinear', align_corners=False)
    up = up.view(m, 4, 3, H // 4, W).permute(0, 2, 3, 1, 4).reshape(m, 3, H, W)
    return up[:, 2], up[:, 0], up[:, 1]


def _held_matrix(mx, H):
    """mx (m, 4, 4): per frame and frame-row class a 2x2 chroma matrix (a, b, c, d) minus
    identity, zero mean over frames and classes -> per-row coefficient maps (m, H, 1) each.
    Real discs: each frame's composite carries a line-alternating chroma crosstalk (V into
    the U carrier ~ +-0.3, gain errors ~0.2, sign flipping frame to frame) that one exact
    carrier model cannot represent for all frames at once."""
    m = mx.shape[0]
    mx = (mx - mx.mean((0, 1), keepdim=True)) / 256
    q = torch.arange(H, device=mx.device) % 4
    rows = mx[:, q]                                  # (m, H, 4)
    return tuple(rows[..., k:k + 1] for k in range(4))


def _held_loss(y, cb, cr, dc, dl, mx, C, cu, cv, A, wf, AY, B0, lam_y, eps_y, lam_cx, lam_cy, eps_c, lam_d):
    """shared latent (H, W) against m frames' composites (m, H, W), each frame's model
    with its own coarse mismatch corrections (_held_mismatch); data residual of frame i
    weighted per pixel by A[i] before the per-row noise-weighted spectrum"""
    H, W = y.shape
    dy, dcb, dcr = _held_mismatch(dc, dl, H, W)
    Y, Cb, Cr = y * 256, cb * 256, cr * 256
    a, b, c, d = _held_matrix(mx, H)
    Cbi = Cb + a * Cb + b * Cr + dcb * 256
    Cri = Cr + c * Cb + d * Cr + dcr * 256
    r = (AY * (Y + dy * 256) + B0 + Cbi * cu + Cri * cv - C) * A
    R = torch.fft.rfft(r, dim=-1)
    loss = ((R.real ** 2 + R.imag ** 2) * wf).sum() * (2.0 / r.shape[-1])
    gx = Y[..., :, 1:] - Y[..., :, :-1]; gy = Y[..., 1:, :] - Y[..., :-1, :]
    loss = loss + lam_y * eps_y * (torch.sqrt(gx ** 2 + eps_y ** 2).sum() + torch.sqrt(gy ** 2 + eps_y ** 2).sum())
    for c in (Cb, Cr):
        dx = c[..., :, 1:] - c[..., :, :-1]; dy_ = c[..., 1:, :] - c[..., :-1, :]
        loss = loss + eps_c * (lam_cx * torch.sqrt(dx ** 2 + eps_c ** 2).sum() + lam_cy * torch.sqrt(dy_ ** 2 + eps_c ** 2).sum())
    loss = loss + lam_d * ((dc * 256) ** 2).sum() + lam_d * ((dl * 256) ** 2).sum() + 1e3 * lam_d * ((mx * 256) ** 2).sum()
    return loss / 65536.0


_held_loss_c = None


def held_static(Y):
    """Y (m, H, W): the run's per-frame luma (solutions).  True when the run does not move.
    Each frame pair's MEDIAN block-matching flow vector is ~0.01-0.03 px on real static
    runs (South 1600-1631, 45026-45057, test card); a pan moves all blocks the same way.
    Tested at lag 1 and at lag m/2, so slow drifts that would blur when averaged
    (bench: 0.05-0.1 px/frame pans) are caught: both medians must stay below HELD_FLOW."""
    Y = np.asarray(Y, np.float32)
    if os.environ.get('PALDEC_HELD_DUMP'):          # calibration: keep the run's luma
        import glob as _g
        np.save(f"{os.environ['PALDEC_HELD_DUMP']}_{len(_g.glob(os.environ['PALDEC_HELD_DUMP'] + '_*'))}.npy", Y.astype(np.uint16))
    worst = 0.0
    # 2026-10-27 (herring integration): aperture.  A picture with structure in one direction only (colour bars: vertical
    # edges, no vertical detail) gives no measurable motion along the other; phase correlation returns noise there
    # (North bars 3027-3046: median dy 0.6-0.7 px, dx 0.01 -> "moving", no held averaging, grain 18.6 vs 5.8).  When
    # vertical carries < HELD_APERTURE of the frame's gradient energy, the vertical flow component is not tested.
    # Measured on the frame-line-pair mean (the fields' own difference is not vertical structure), sigma 2 smoothed:
    # bars 0.03, test card / photos / maps / text / video 0.5-0.97.  Only the vertical component is exempted.
    ycomp = None
    if HELD_APERTURE > 0:
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(0.5 * (Y[0][0::2] + Y[0][1::2]), 2.0)[8:-8, 8:-8]
        gx = float((np.diff(Z, axis=1) ** 2).sum()); gy = 4 * float((np.diff(Z, axis=0) ** 2).sum())
        HELD_DEBUG['gy_frac'] = gy / (gx + gy + 1e-9)
        if gy < HELD_APERTURE * (gx + gy): ycomp = 1
    for L in sorted({1, max(1, len(Y) // 2)}):
        if len(Y) <= L:
            continue
        f = estimate_flow_gpu(Y, lag=L)[L:]
        v = np.median(f.reshape(len(f), -1, 2), axis=1)
        if ycomp is not None:
            v[:, ycomp] = 0
        worst = max(worst, float(np.median(np.hypot(v[:, 0], v[:, 1]))))
    HELD_DEBUG['flow'] = worst
    # 2026-10-28 (herring integration): cumulative drift.  A slow pan (NationalB 69431-69460, ~0.2 px/frame) can pass
    # the per-pair medians on noisy solutions yet moves px over the run; averaged as a held still it blurs and its
    # outlier blocks keep per-frame noise (motion combing x1.19).  Displacement first -> each later frame, linear fit
    # through 0: slope x (m - 1) is the run's drift; a true still drifts ~0.
    HELD_DEBUG['drift'] = 0.0
    if HELD_DRIFT > 0 and len(Y) >= 3:
        d = []
        for j0 in range(1, len(Y), 16):                                          # pairs (first, j): flow at odd entries
            f = estimate_flow_gpu(np.stack([Y[k] for j in range(j0, min(len(Y), j0 + 16)) for k in (0, j)]), lag=1)[1::2]
            d.append(np.median(f.reshape(len(f), -1, 2), axis=1))
        d = np.concatenate(d).astype(np.float64)
        if ycomp is not None:
            d[:, ycomp] = 0
        t = np.arange(1, len(Y), dtype=np.float64)
        sl = (t[:, None] * d).sum(0) / (t * t).sum()
        HELD_DEBUG['drift'] = float(np.hypot(*sl) * (len(Y) - 1))
    # local motion on a static background (NationalB 69442 figures in an inset, Countryside / City_Financial talking
    # heads) leaves the median flow and the drift ~0; its 32 px blocks differ first -> last frame by > HELD_MOVK x the
    # picture's median block difference (noise).  Fraction of such blocks, quickcheck panels' held candidates: true
    # stills (bars, test card, maps, menus, UN / South / north2 held runs) 0.000-0.008, local motion 0.03-0.20.
    from scipy.ndimage import gaussian_filter
    D = np.abs(gaussian_filter(Y[-1], 1.0) - gaussian_filter(Y[0], 1.0))[8:-8, 8:-8]
    hb, wb = D.shape[0] // 32, D.shape[1] // 32
    b = D[:hb * 32, :wb * 32].reshape(hb, 32, wb, 32).mean((1, 3))
    HELD_DEBUG['movfrac'] = float((b > HELD_MOVK * np.median(b)).mean())
    return worst < HELD_FLOW and HELD_DEBUG['drift'] < (HELD_DRIFT if HELD_DRIFT > 0 else np.inf) \
        and HELD_DEBUG['movfrac'] < (HELD_MOVFRAC if HELD_MOVFRAC > 0 else np.inf)


def solve_held(C, cu, cv, X, noise_w, still, hanover=0., iters=None, hanover_iters=30, lr=1e4, dmask=None, own=None):
    """C, cu, cv: (m, H, W) torch for the m frames of a static run; X (m, 3, H, W) numpy:
    their per-frame solutions.  Returns (m, 3, H, W) numpy: the shared latent where each
    frame agrees with the consensus, its own solution elsewhere.
    Real discs: the frames' composites are NOT all explained by one picture through the
    exact carrier model -- each frame carries a line-alternating chroma error (~10% of the
    chroma, Hanover-type, sign flipping frame to frame) and small level drifts, which a
    single-frame solve leaves in its residual (stage-2 prior) but which a shared latent
    would have to absorb as checkerboard.  So each frame gets coarse (HELD_DGRID px)
    per-row-class chroma and luma corrections in its data term (_held_mismatch); the
    output is the shared latent itself.
    1. consensus Z0 = per-pixel median of X; gross outlier blocks (T x HELD_T1)
    2. shared latent + per-frame corrections from all frames' data terms
    3. outliers re-tested against each frame's corrected model (HELD_T); re-solve
    4. per frame: latent where it agrees, its own solution in its outlier blocks"""
    import torch.nn.functional as Fn
    global _held_loss_c
    iters = iters or HELD_ITERS
    m, _, H, W = np.shape(X)
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev)
    Dm = _dm(dmask, C)
    Z0 = Xt.median(0).values
    out = held_outliers(Z0, C, cu, cv, T=HELD_T * HELD_T1, Dm=Dm)
    A = held_alpha(out, H, W)
    cnt = A.sum(0)
    Zi = (Xt * A[:, None]).sum(0) / cnt.clamp(min=1e-3)[None]
    Zi = torch.where(cnt[None] > 1e-3, Zi, Z0)
    del Z0
    y = (Zi[0] / 256).clone().requires_grad_(True)
    cb = ((Zi[1] - 32768) / 256).clone().requires_grad_(True)
    cr = ((Zi[2] - 32768) / 256).clone().requires_grad_(True)
    del Zi
    g = HELD_DGRID or 32
    gy, gx = max(1, H // 4 // (g // 4)), max(1, W // g)
    dc = torch.zeros((m, 8, gy, gx), device=dev, requires_grad=True)
    dl = torch.zeros((m, 4, gy, gx), device=dev, requires_grad=True)
    mx = torch.zeros((m, 4, 4), device=dev, requires_grad=True)
    params = [y, cb, cr] + ([dc, dl] if HELD_DGRID else []) + ([mx] if HELD_MATRIX else [])
    lam_d = HELD_LAMD if HELD_DGRID else 1e6
    wf = torch.tensor(noise_w, device=dev)
    if _held_loss_c is None:
        _held_loss_c = torch.compile(_held_loss, dynamic=True)
    # priors scale with the run length: the data term is the sum of m frames' terms, and on
    # real discs a prior that does not grow with it lets the latent follow the data into
    # bands the single-frame priors keep quiet (split-half noise test)
    ps = HELD_PS * max(1.0, m / 8.0) ** 0.5
    consts = (float(M.P['AY']), float(M.P['B']), float(still['lam_y']) * ps, float(still['eps_y']),
              float(still['lam_cx']) * ps * HELD_PSC, float(still['lam_cy']) * ps * HELD_PSC, float(still['eps_c']), float(lam_d))
    Ab = [A]
    objective = lambda: _held_loss_c(y, cb, cr, dc, dl, mx, C, cu, cv, Ab[0] * Dm, wf, *consts)
    def run(n, ps):
        opt = torch.optim.LBFGS(ps, lr=lr, max_iter=n, history_size=HIST,
                                line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
        def closure():
            opt.zero_grad(); l = objective(); l.backward(); return l
        opt.step(closure)
    run(iters, params)
    with torch.no_grad():
        dy, dcb, dcr = _held_mismatch(dc, dl, H, W)
        a, b, c, d = _held_matrix(mx, H)
        if not HELD_RETEST_D:            # the coarse per-frame offsets can absorb a small local change (cursor)
            dy = dy * 0; dcb = dcb * 0; dcr = dcr * 0
        Zf = torch.stack([y * 256 + dy * 256, (cb + a * cb + b * cr + dcb) * 256 + 32768, (cr + c * cb + d * cr + dcr) * 256 + 32768])
        del dy, dcb, dcr, a, b, c, d
    out2 = held_outliers(Zf, C, cu, cv, Dm=Dm)
    del Zf
    HELD_DEBUG['resolve'] = int((out2 != out).sum())
    out2 = held_decoded_check(Xt, out2)
    Ab[0] = A = held_alpha(out2, H, W)
    if HELD_MODE == 'joint':             # avg mode needs the latent only for the outlier test
        run(max(10, iters // 2), params)
        if hanover:
            _hanover_stage(y, cb, cr, objective, hanover, hanover_iters, None, lr)
    with torch.no_grad():
        HELD_DEBUG['dc_rms'] = float(_held_mismatch(dc, dl, H, W)[1].pow(2).mean().sqrt()) if HELD_DGRID else 0.
        HELD_DEBUG['mx'] = ((mx - mx.mean((0, 1), keepdim=True)) / 256).cpu().numpy()
    with torch.no_grad():
        if HELD_MODE == 'joint':
            Z = torch.stack([y * 256, cb * 256 + 32768, cr * 256 + 32768])
        else:                            # 'avg': inlier-weighted mean of the per-frame solutions
            Z = (Xt * A[:, None]).sum(0) / A.sum(0).clamp(min=1e-3)[None]
            if dmask is not None and HELD_DOW:
                # v6: a frame's dropout pixels (its solve filled them from the priors) get its dropout-mask
                # weight, so they come from the run's other frames; where every inlier has one, as v5
                Aw = A * Dm
                Zw = (Xt * Aw[:, None]).sum(0) / Aw.sum(0).clamp(min=1e-3)[None]
                Z = torch.where(Aw.sum(0)[None] > 1e-3, Zw, Z)
                del Aw, Zw
        if own is not None:              # outlier blocks take these per-frame solutions instead (local-motion runs: video)
            Xt = torch.as_tensor(np.asarray(own, np.float32), device=dev)
        o = A[:, None] * Z[None] + (1 - A[:, None]) * Xt
    return o.cpu().numpy()
