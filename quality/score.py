# SPDX-License-Identifier: GPL-3.0-or-later
"""Score an fq benchmark run against the known truth, per frame (fq benchmark, 2026-10-24).

  score.py <benchdir> <label> [<label2> ...]     score each label; with 2+ labels also print the change

Two levels, both full-reference (the truth is exact, so nothing here is an estimate):

  composite (stack.tbc and doc.tbc vs clean.tbc, per field; stacker / dropout-correct quality):
      csnr_<band>   signal-to-error ratio of the composite in a band (dB): 10 log10(sum|T|^2 / sum|E|^2)
                    bands 0-2, 2-3.5, 3.5-5.5 (chroma), 5.5-8.9 MHz; active area only
  picture (out.yuv vs truth.yuv, per frame; whole chain incl. the decoder):
      psnr_y / psnr_cb / psnr_cr   PSNR, peak = nominal white-black (219 x 256)
      rmse_y8                      luma RMS error in 8-bit units
      ssim_y                       SSIM of luma (Gaussian sigma 1.5, 8-bit scale)
      snr_y_<band>                 luma signal-to-error ratio per horizontal band (dB), 0-1 ... 4-5 MHz
      gain_y_<band>                detail retention: Re<O,T>/<T,T> per band (1 = kept; < 1 = softened)
      snr_c_<band>, gain_c_<band>  the same for chroma (Cb+Cr), 0-0.5, 0.5-1, 1-1.5 MHz
      noise_y8 / noise_c8          output noise (8-bit rms): the error minus its Gaussian blur (sigma 2 px), over
                                   the pixels where the truth is locally flat (7x7 luma std < 1.5 levels); 2026-10-26
  The sub-pixel shift between output and truth is measured per field / frame and removed before scoring; it
  is reported (dx), and its spread over the run as jitter_dx_*.


Writes <seg>/<label>/score_frames.csv and score_fields.csv, and <benchdir>/<label>.summary.txt.
"""
import sys, os, json, csv
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

sys.path.insert(0, os.environ.get('LD', os.path.expanduser('~/ld-decode')))
from lddecode.stack import cav_picture, field_vbi

W, H, FS = 1135, 313, 17734475.0
FW, FH = 928, 576
PEAK = 219 * 256.0
YB = [(0, 1e6), (1e6, 2e6), (2e6, 3e6), (3e6, 4e6), (4e6, 5e6)]
CB = [(0, 0.5e6), (0.5e6, 1e6), (1e6, 1.5e6)]
TB = [(0, 2e6), (2e6, 3.5e6), (3.5e6, 5.5e6), (5.5e6, FS / 2)]
R0, R1, C0, C1 = 8, FH - 8, 16, FW - 16          # picture crop
GAIN_MIN_RMS = 300.0     # truth rms in a band (16-bit) below which gain / SNR of that band are not scored
TR0, TR1, TC0, TC1 = 22, 309, 185, 1107           # composite active area


def bname(lo, hi):
    return f'{lo / 1e6:g}-{hi / 1e6:.1f}'


def fshift(img, dy, dx):
    Fy = np.fft.fftfreq(img.shape[-2])[:, None]; Fx = np.fft.fftfreq(img.shape[-1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (Fy * dy + Fx * dx))))


def est_shift(a, b):
    """Sub-pixel (dy, dx) with b(x) ~ a(x - d): phase correlation + parabolic peak."""
    A = np.fft.fft2(a - a.mean()); Bf = np.fft.fft2(b - b.mean())
    r = np.real(np.fft.ifft2(Bf * np.conj(A) / (np.abs(Bf * np.conj(A)) + 1e-9)))
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    def par(m, c, p):
        d = m - 2 * c + p
        return 0.0 if d == 0 else 0.5 * (m - p) / d
    dy = iy + par(r[iy - 1, ix], r[iy, ix], r[(iy + 1) % r.shape[0], ix])
    dx = ix + par(r[iy, ix - 1], r[iy, ix], r[iy, (ix + 1) % r.shape[1]])
    if dy > r.shape[0] / 2: dy -= r.shape[0]
    if dx > r.shape[1] / 2: dx -= r.shape[1]
    return dy, dx


def band_stats(T, O, bands, fs, err=False):
    """Per band: SNR dB of O vs T, gain Re<O,T>/<T,T> (and with err=True the error rms in the band, input
    units).  Row-wise rfft along x."""
    f = np.fft.rfftfreq(T.shape[-1], 1 / fs)
    win = np.hanning(T.shape[-1])
    Ft = np.fft.rfft((T - T.mean(-1, keepdims=True)) * win, axis=-1)
    Fo = np.fft.rfft((O - O.mean(-1, keepdims=True)) * win, axis=-1)
    Fe = Fo - Ft
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        pt = np.sum(np.abs(Ft[..., m]) ** 2); pe = np.sum(np.abs(Fe[..., m]) ** 2)
        g = np.sum((Fo[..., m] * np.conj(Ft[..., m])).real) / max(pt, 1e-9)
        r = (10 * np.log10(max(pt, 1e-9) / max(pe, 1e-9)), g)
        if err:   # Parseval with the Hann window: sum|F|^2 = N * sum(w^2) * var over the band
            r += (float(np.sqrt(2 * pe / (T.shape[-1] * (win ** 2).sum() * np.prod(T.shape[:-1])))),)
        out.append(r)
    return out


def ssim(a, b):
    a = a / 256.0; b = b / 256.0
    C1_, C2_ = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = gaussian_filter(a, 1.5); mu_b = gaussian_filter(b, 1.5)
    saa = gaussian_filter(a * a, 1.5) - mu_a ** 2; sbb = gaussian_filter(b * b, 1.5) - mu_b ** 2
    sab = gaussian_filter(a * b, 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1_) * (2 * sab + C2_)) / ((mu_a ** 2 + mu_b ** 2 + C1_) * (saa + sbb + C2_))
    return float(m.mean())


def pics(js):
    """(CAV picture, first field?) -> field index; second fields carry no picture number, they inherit the
    preceding first field's."""
    out = {}; last = None
    for k, fj in enumerate(js['fields']):
        first = bool(fj.get('isFirstField', k % 2 == 0))
        p = cav_picture(field_vbi(fj))
        if first:
            last = p
        elif p is None:
            p = last
        if p is not None:
            out[(p, first)] = k
    return out


def score_composite(d, o, which):
    t = f'{o}/{which}.tbc'
    if not os.path.exists(t):
        return []
    cl = np.memmap(f'{d}/clean.tbc', dtype='<u2', mode='r').reshape(-1, H, W)
    x = np.memmap(t, dtype='<u2', mode='r').reshape(-1, H, W)
    pc = pics(json.load(open(f'{d}/clean.tbc.json'))); px = pics(json.load(open(t + '.json')))
    keys = sorted(set(pc) & set(px))
    if not keys:
        return []
    rows = []
    for key in keys:          # shift measured per field (and reported: a varying dx is jitter)
        T = cl[pc[key]][TR0:TR1, TC0:TC1].astype(float)
        dy, dx = est_shift(T, x[px[key]][TR0:TR1, TC0:TC1].astype(float))
        X = fshift(x[px[key]].astype(float), 0, -dx)[TR0:TR1, TC0:TC1]
        bs = band_stats(T, X, TB, FS, err=True)
        rows.append({'picture': key[0], 'field': 1 if key[1] else 2, 'dx': round(dx, 3),
                     **{f'{which}_csnr_{bname(*b)}': round(v[0], 2) for b, v in zip(TB, bs)},
                     **{f'{which}_err_{bname(*b)}': round(v[2], 1) for b, v in zip(TB, bs)},
                     f'{which}_err_rms': round(float(np.sqrt(np.mean((X - T) ** 2))), 1)})
    return rows


def score_picture(d, o, start):
    p = f'{o}/out.yuv'
    if not os.path.exists(p):
        return []
    T = np.memmap(f'{d}/truth.yuv', dtype='<u2', mode='r').reshape(-1, 3, FH, FW)
    O = np.memmap(p, dtype='<u2', mode='r').reshape(-1, 3, FH, FW)
    n = min(len(T), len(O))
    rows = []
    for k in range(n):        # shift measured per frame (reported; its spread is frame-to-frame jitter)
        dy, dx = est_shift(T[k, 0].astype(float), O[k, 0].astype(float))
        t = T[k].astype(float); o_ = np.stack([fshift(O[k, c].astype(float), -dy, -dx) for c in range(3)])
        t = t[:, R0:R1, C0:C1]; o_ = o_[:, R0:R1, C0:C1]
        e = o_ - t
        mse = (e ** 2).mean(axis=(1, 2))
        r = {'frame': start + k, 'dy': round(dy, 3), 'dx': round(dx, 3)}
        for c, nm in enumerate(('y', 'cb', 'cr')):
            r[f'psnr_{nm}'] = round(10 * np.log10(PEAK ** 2 / max(mse[c], 1e-9)), 2)
        r['rmse_y8'] = round(float(np.sqrt(mse[0])) / 256, 3)
        r['ssim_y'] = round(ssim(t[0], o_[0]), 4)
        mu = uniform_filter(t[0], 7); flat = (uniform_filter(t[0] ** 2, 7) - mu ** 2) < 384.0 ** 2
        flat[:4] = flat[-4:] = False; flat[:, :4] = flat[:, -4:] = False
        if flat.sum() > 2000:
            hp = [e[c] - gaussian_filter(e[c], 2) for c in range(3)]
            r['noise_y8'] = round(float(np.sqrt((hp[0][flat] ** 2).mean())) / 256, 3)
            r['noise_c8'] = round(float(np.sqrt(((hp[1][flat] ** 2).mean() + (hp[2][flat] ** 2).mean()) / 2)) / 256, 3)
        else:
            r['noise_y8'] = r['noise_c8'] = float('nan')
        # gain / SNR only where the truth has content in the band (a near-black frame makes them noise / ~0);
        # err (rms of the error in the band, 16-bit units) is content-independent and always reported
        for nm, T_, O_, bands in (('y', t[0], o_[0], YB),
                                  ('c', np.concatenate([t[1], t[2]]), np.concatenate([o_[1], o_[2]]), CB)):
            for b, (sn, g, er), tr in zip(bands, band_stats(T_, O_, bands, FS, err=True),
                                          band_stats(T_, np.zeros_like(T_), bands, FS, err=True)):
                ok = tr[2] > GAIN_MIN_RMS       # truth rms in the band (error vs zero)
                r[f'err_{nm}_{bname(*b)}'] = round(er, 1)
                r[f'snr_{nm}_{bname(*b)}'] = round(sn, 2) if ok else float('nan')
                r[f'gain_{nm}_{bname(*b)}'] = round(g, 3) if ok else float('nan')
        rows.append(r)
    return rows


def write_csv(path, rows):
    if rows:
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def summarise(rows, skip=('frame', 'picture', 'field', 'dy', 'dx')):
    if not rows:
        return {}
    return {k: float(np.nanmean([r[k] for r in rows])) if np.isfinite([r[k] for r in rows]).any() else float('nan')
            for k in rows[0] if k not in skip}


def main():
    bench, labels = sys.argv[1], sys.argv[2:]
    meta = json.load(open(f'{bench}/bench.json'))
    allsum = {}
    for lab in labels:
        segsum = {}; pf = []; cf = []
        for sg in meta['segments']:
            d = sg['dir']; o = f'{d}/{lab}'
            fr = score_picture(d, o, sg['start'])
            fl = score_composite(d, o, 'stack') ; dl = score_composite(d, o, 'doc')
            for a, b in zip(fl, dl):
                a.update({k: v for k, v in b.items() if k.startswith('doc_')})
            fl = fl or dl
            write_csv(f'{o}/score_frames.csv', fr); write_csv(f'{o}/score_fields.csv', fl)
            segsum[sg['start']] = {**summarise(fr), **summarise(fl)}
            pf += fr; cf += fl
        tot = {**summarise(pf), **summarise(cf)}
        if pf: tot['jitter_dx_frames'] = float(np.std([r['dx'] for r in pf]))
        if cf: tot['jitter_dx_fields'] = float(np.std([r['dx'] for r in cf]))
        allsum[lab] = (tot, segsum)
        keys = list(tot)
        with open(f'{bench}/{lab}.summary.txt', 'w') as fh:
            fh.write(f'{lab}: {len(pf)} frames, {len(cf)} fields\n')
            fh.write(f'{"metric":22s} {"all":>8s} ' + ' '.join(f'{s:>8d}' for s in segsum) + '\n')
            for k in keys:
                fh.write(f'{k:22s} {tot[k]:8.3f} ' + ' '.join(f'{segsum[s].get(k, float("nan")):8.3f}' for s in segsum) + '\n')
        print(open(f'{bench}/{lab}.summary.txt').read())
    if len(labels) >= 2:
        a, b = labels[0], labels[-1]
        ta, sa = allsum[a]; tb, sb = allsum[b]
        print(f'change {a} -> {b} (mean; per-segment changes; dB metrics: + is better)')
        for k in ta:
            if k in tb:
                per = [sb[s].get(k, np.nan) - sa[s].get(k, np.nan) for s in sa]
                print(f'  {k:22s} {tb[k] - ta[k]:+8.3f}   ' + ' '.join(f'{v:+7.3f}' for v in per))


if __name__ == '__main__':
    main()
