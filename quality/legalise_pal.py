"""Make a 928x576 yuv444p16le truth PAL-legal for the fq benchmark: horizontal low-pass of luma at 5.0 MHz and
chroma at 1.3 MHz (the PAL I system bandwidths, at the 17.73 MHz 4fsc sample rate the decoders output at),
and Y / Cb / Cr clipped to the legal 16-235 / 16-240 range.  Content beyond these is the same loss for every
decoder and would only dilute the comparison.
  legalise_pal.py <in.yuv> <out.yuv>"""
import sys, numpy as np
W, H, FS = 928, 576, 17734475.0
f = np.fft.rfftfreq(W, 1 / FS)
def lp(fc, tw=0.4e6):                      # raised-cosine roll-off fc-tw .. fc+tw
    return np.clip(0.5 + 0.5 * np.cos(np.pi * np.clip((f - (fc - tw)) / (2 * tw), 0, 1)), 0, 1)
HY, HC = lp(5.0e6), lp(1.3e6, 0.3e6)
x = np.fromfile(sys.argv[1], '<u2').reshape(-1, 3, H, W).astype(np.float64)
for c, h in ((0, HY), (1, HC), (2, HC)):
    m = x[:, c].mean(-1, keepdims=True)
    x[:, c] = np.fft.irfft(np.fft.rfft(x[:, c] - m, axis=-1) * h, n=W, axis=-1) + m
x[:, 0] = np.clip(x[:, 0], 16 * 256, 235 * 256); x[:, 1:] = np.clip(x[:, 1:], 16 * 256, 240 * 256)
np.rint(x).astype('<u2').tofile(sys.argv[2])
print('legalised', len(x), 'frames ->', sys.argv[2])
