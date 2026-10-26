#!/usr/bin/env python3
"""Probe .ldf captures for the LD-V4300D ~8.47/8.56 MHz CW spur.
Reads a mid-disc RF window, Welch-PSDs it, and reports the peak-over-floor
ratio in 8.30-8.70 MHz. Prints: name  peak_dB  freq_MHz  verdict."""
import sys, numpy as np
import os
sys.path.insert(0, os.environ.get("LD", os.path.expanduser("~/ld-decode")))
from lddecode.utils import make_loader

N = 1 << 22          # ~4.2M samples = 0.105 s @ 40 MHz
OFFSETS = (5_000_000_000, 20_000_000_000, 60_000_000_000)  # the V4300D spur is
# strongest on CAV inner tracks and fades with radius: report the MAX over
# start/mid/late windows (a single mid-disc probe under-reads by 10+ dB)

for path in sys.argv[1:]:
    name = path.split("/")[-1]
    try:
        loader = make_loader(path, None)
        db, fpk = -1.0, 0.0
        for off in OFFSETS:
            try:
                data = loader(open(path, "rb"), off, N)
            except Exception:
                continue
            if data is None:
                continue
            x = np.asarray(data, np.float64)
            if x.ndim == 0 or len(x) < 65536:
                continue
            x -= x.mean()
            seg = 65536
            nseg = len(x) // seg
            w = np.hanning(seg)
            psd = np.zeros(seg // 2 + 1)
            for i in range(nseg):
                psd += np.abs(np.fft.rfft(x[i*seg:(i+1)*seg] * w)) ** 2
            psd /= nseg
            freqs = np.fft.rfftfreq(seg, 1 / 40e6)
            band = (freqs >= 8.30e6) & (freqs <= 8.70e6)
            d = 10 * np.log10(psd[band].max() / np.median(psd[band]))
            if d > db:
                db, fpk = d, freqs[band][np.argmax(psd[band])]
        if db < 0:
            print(f"{name}  ERROR no usable window (every offset past EOF?)", flush=True)
            continue
        verdict = "SPUR" if db > 12 else "clean"
        print(f"{name}  {db:6.1f} dB  {fpk/1e6:.3f} MHz  {verdict}", flush=True)
    except Exception as e:
        print(f"{name}  ERROR {e}", flush=True)
