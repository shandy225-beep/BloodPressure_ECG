"""
Filter bandpass IIR Chebyshev Type II untuk sinyal ECG.

# Persamaan 2.x proposal — Preprocessing sinyal ECG (bandpass 2-40 Hz)

Fungsi di modul ini dipakai IDENTIK oleh training pipeline (src/data_pipeline)
maupun real-time pipeline (src/gui, src/acquisition) — tidak ada duplikasi
logic filter di tempat lain.
"""

import numpy as np
from scipy.signal import cheby2, filtfilt

from src.config import (
    FS_TARGET,
    BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ,
    BANDPASS_ORDER,
    BANDPASS_RS_DB,
)


def apply_bandpass_cheby2(
    signal: np.ndarray,
    fs: int = FS_TARGET,
    low: float = BANDPASS_LOW_HZ,
    high: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
    rs: float = BANDPASS_RS_DB,
) -> np.ndarray:
    """Bandpass filter Chebyshev Type II, zero-phase (filtfilt).

    Parameter default: order=4, rs=40 dB, band 2-40 Hz relatif terhadap
    fs=FS_TARGET (Nyquist = fs/2). Order 4 dipilih karena cukup tajam untuk
    membuang baseline wander (<2 Hz) dan noise otot/powerline (>40 Hz) tanpa
    mendistorsi kompleks QRS secara berlebihan; rs=40 dB adalah nilai umum
    untuk stopband attenuation Chebyshev II pada aplikasi biosignal.

    filtfilt (bukan lfilter) dipakai supaya filter zero-phase — tidak ada
    pergeseran waktu (delay) pada morfologi gelombang ECG, penting karena
    fitur non-fiducial di modul ini bergantung pada bentuk sinyal utuh.
    """
    signal = np.asarray(signal, dtype=float)
    nyquist = fs / 2.0
    low_norm = low / nyquist
    high_norm = high / nyquist

    b, a = cheby2(order, rs, [low_norm, high_norm], btype="bandpass")

    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(signal) <= padlen:
        # Sinyal terlalu pendek untuk padding default filtfilt -> kurangi padlen
        padlen = max(0, len(signal) - 1)

    filtered = filtfilt(b, a, signal, padlen=padlen if padlen > 0 else None)
    return filtered
