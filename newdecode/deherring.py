"""Remove the disc's FM-beat herringbone from TBC fields.

Community South carries a beat between the FM video carrier and the colour
signal that was recorded on the master (it survives a 9-capture stack): in
strongly coloured flat areas a faint (~1 level peak, 8-bit) slanted plane
wave at horizontal frequency
    f = F0 + K * brightness   (F0 2.232 MHz, K 2.34e-5 MHz per 16-bit level;
                               fitted over 300 detections, 97 kHz rms)
- 2.8-3.2 MHz at the usual levels: herringbone in reds and magentas.

It is far below the noise on any one line, so it is estimated coherently
over 2-D tiles (TY field lines x TX samples): each line is demodulated at
the predicted frequency, the tile's line-to-line phase step is found from
the spectrum of those complex amplitudes, and one amplitude/phase is fitted
for the whole tile.  A tile is corrected only when
  * it is strongly coloured (the beat needs colour),
  * the plane wave holds most of the demodulated energy across its lines
    (coherent, not noise or texture),
  * its line-to-line step is not ~0 (Test Card F's gratings and other
    vertical structure are line-locked),
  * the picture is flat there (<1.5 MHz luma std under FLAT_MAX) and the
    wave is faint (under AMP_MAX) - textured areas can contain a
    coincidentally coherent slanted component that is real detail.
Correction is the fitted wave times a 2-D Hann, overlap-added (50%), so
tiles that fail the tests are left exactly as they were."""
import numpy as np

FS = 17.734475
F0, K = 2.232, 2.34e-5
TX, TY = 64, 32
CHROMA_MIN = 15000.0
COH_MIN = 0.35        # fraction of the tile's demodulated energy in the plane wave
FLAT_MAX = 3.0 * 256  # 16-bit std of the tile's <1.5 MHz luma: herringbone only where the picture is flat
AMP_MAX = 4.0 * 256   # the beat is a faint 1-3 levels; a stronger 'wave' is picture texture
STEP_MIN = 0.08       # cycles per field line: not line-locked
X0, X1 = 186, 1106
_wx = np.hanning(TX); _wy = np.hanning(TY)
_fq = np.fft.rfftfreq(256, 1 / FS); _chr = (_fq > 3.9) & (_fq < 5.0)


def deherring_field(field, l0=24, l1=308):
    f = field.astype(np.float64)
    corr = np.zeros_like(f); n = 0
    xs = np.arange(TX)
    for y in range(l0, l1 - TY + 1, TY // 2):
        for x in range(X0, X1 - TX + 1, TX // 2):
            t = f[y:y + TY, x:x + TX]
            lvl = t.mean(); d = t - t.mean(1, keepdims=True)
            # flat-area gate (frame 16275's dense floral texture once lost
            # 21 levels of real detail to a coincidentally coherent 'wave')
            lo = np.fft.irfft(np.fft.rfft(t, axis=1) * (np.fft.rfftfreq(TX, 1 / FS) < 1.5), TX, axis=1)
            if lo.std() > FLAT_MAX:
                continue
            P = (np.abs(np.fft.rfft(d * _wx, 256, axis=1)) ** 2).mean(0)
            chroma = np.sqrt(P[_chr].sum()) * 2 / _wx.sum()
            if chroma < CHROMA_MIN:
                continue
            fp = F0 + K * lvl
            e = np.exp(-2j * np.pi * fp / FS * xs)
            z = (d * _wx * e).sum(1) / _wx.sum() * 2           # complex amplitude per line
            Z = np.fft.fft(z * _wy, 512)
            k = int(np.argmax(np.abs(Z))); fy = np.fft.fftfreq(512)[k]
            if abs(fy) < STEP_MIN:
                continue
            ly = np.arange(TY)
            A = (z * np.exp(-2j * np.pi * fy * ly) * _wy).sum() / _wy.sum()
            model = A * np.exp(2j * np.pi * fy * ly)
            coh = (np.abs(model) ** 2 * _wy).sum() / ((np.abs(z) ** 2 * _wy).sum() + 1e-9)
            if coh < COH_MIN or abs(A) > AMP_MAX:
                continue
            wave = np.real(model[:, None] * np.exp(2j * np.pi * fp / FS * xs)[None, :])
            corr[y:y + TY, x:x + TX] += wave * _wy[:, None] * _wx[None, :]
            n += 1
    return f - corr, n
