"""Streaming sample-rate conversion to 16 kHz, numpy only (no scipy).

Microphones usually run at 44.1 / 48 kHz (AirPods: 24 kHz). Each audio callback
gives a block; blocks are converted with state carried across them, so there
are no clicks at block boundaries:

  1. low-pass FIR (windowed sinc) at ~90 % of the output Nyquist — anti-aliasing
  2. linear interpolation at the output sample times

After a proper low-pass, linear interpolation is plenty for speech recognition,
and the whole thing costs well under 0.1 ms per 32 ms block.
"""
from __future__ import annotations

import numpy as np


class StreamResampler:
    def __init__(self, rate_in: int, rate_out: int = 16000, taps: int = 63):
        self.rate_in, self.rate_out = rate_in, rate_out
        self.step = rate_in / rate_out  # input samples per output sample
        self.passthrough = rate_in == rate_out
        if rate_in > rate_out:
            fc = 0.45 * rate_out / rate_in  # cutoff, cycles per input sample
            n = np.arange(taps) - (taps - 1) / 2
            h = 2 * fc * np.sinc(2 * fc * n) * np.blackman(taps)
            self.h = (h / h.sum()).astype(np.float32)
        else:
            self.h = np.ones(1, np.float32)  # upsampling: nothing to remove
        self.hist = np.zeros(len(self.h) - 1, np.float32)  # filter memory
        self.buf = np.zeros(1, np.float32)  # last filtered sample (interp anchor)
        # next output position (input samples, relative to buf[0]); starting past the
        # FIR's group delay keeps output sample k aligned with input time k*step
        self.t = 1 + (len(self.h) - 1) / 2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.passthrough:
            return x
        padded = np.concatenate([self.hist, x])
        filtered = np.convolve(padded, self.h, mode="valid")  # len(x) samples
        self.hist = padded[len(padded) - len(self.hist):] if len(self.hist) else self.hist
        buf = np.concatenate([self.buf, filtered])
        last = len(buf) - 1
        if self.t > last:
            out = np.empty(0, np.float32)
        else:
            pos = np.arange(self.t, last, self.step)
            out = np.interp(pos, np.arange(len(buf)), buf).astype(np.float32)
            self.t = pos[-1] + self.step if len(pos) else self.t
        self.t -= last
        self.buf = buf[-1:]
        return out
