"""Noise spectrum of a stacked TBC, from its blank vertical-interval lines.

Field lines 6-13, samples 200-1080 carry only black level, so their
variation is noise: the disc's own (master) noise plus what is left of the
players' after stacking.  Returns the per-sample noise PSD on the rfft bins
of a 928-sample decoder row (16-bit units^2 per bin, Parseval-consistent
with numpy rfft of a row), averaged over many fields."""
import os, numpy as np

W, H = 1135, 313
FSZ = W * H * 2


def noise_psd(tbc, nfields=400, out_len=928):
    fd = os.open(tbc, os.O_RDONLY)
    n = os.fstat(fd).st_size // FSZ
    acc = None; cnt = 0
    win = np.hanning(880)
    lo, hi = (200, n - 200) if n > 800 else (0, n - 1)     # short clips: use every field
    for fi in np.unique(np.linspace(lo, hi, min(nfields, hi - lo + 1)).astype(int)):
        f = np.frombuffer(os.pread(fd, FSZ, int(fi) * FSZ), '<u2').reshape(H, W).astype(float)
        seg = f[6:14, 200:1080]
        if np.median(seg) - np.median(f[6:14, 10:50]) < 5000:
            continue
        seg = seg - seg.mean(1, keepdims=True)
        P = (np.abs(np.fft.rfft(seg * win, axis=1)) ** 2).mean(0) / (win ** 2).mean()
        acc = P if acc is None else acc + P; cnt += 1
    os.close(fd)
    P = acc / cnt / 880                                  # per-sample variance density
    fi = np.fft.rfftfreq(880); fo = np.fft.rfftfreq(out_len)
    return np.interp(fo, fi, P) * out_len                # rescaled to |rfft|^2 of an out_len row


if __name__ == '__main__':
    import sys
    P = noise_psd(sys.argv[1])
    f = np.fft.rfftfreq(928, 1 / 17.734475)
    np.save(sys.argv[2], P)
    tot = P.sum() * 2 / 928 ** 2
    print('noise std %.1f (16-bit); share above 5.5 MHz %.0f%%' % (np.sqrt(tot * 928), 100 * P[f > 5.5].sum() / P.sum()))
