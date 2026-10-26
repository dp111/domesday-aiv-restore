"""Remove the record/playback channel's slow edge smear ("ghost to the right").

Every edge on the disc - the VITS bar (both directions) and the sync rising
edge - reaches ~88% at once and the last ~12% follows with a ~0.4 us time
constant (Community South: 11.5-12.6%, 395-435 ns).  That is a channel
H(z) = (1-c) + c(1-a)/(1-a z^-1), a = exp(-1/tau), whose exact inverse
    G(z) = (1 - a z^-1) / ((1 - c a) - (1 - c) a z^-1)
is stable (pole at (1-c)a/(1-ca) < 1).  Applied along each TBC line.
c=0.16, tau=9 were tuned to flatten the bar and sync edges (deviation after
the edge 2.4-3.9% -> 1.0-1.4%); the single-exponential fit underestimates them.
"""
import numpy as np
from scipy.signal import lfilter


def deghost(lines, c=0.16, tau=9.0):
    a = np.exp(-1.0 / tau)
    b = [1.0, -a]
    den = [1.0 - c * a, -(1.0 - c) * a]
    x = np.asarray(lines, dtype=np.float64)
    # run over the line sequence as one continuous signal (lines are contiguous in time)
    flat = x.reshape(-1)
    y = lfilter(b, den, flat - flat[0]) + flat[0]
    return y.reshape(x.shape)
