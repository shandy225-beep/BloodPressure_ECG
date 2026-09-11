"""
Filter preprocessing ECG: notch powerline interference + bandpass IIR
Chebyshev Type II.

# Persamaan 2.x proposal — Preprocessing sinyal ECG (notch 50Hz + bandpass 2-40 Hz)

Fungsi di modul ini dipakai IDENTIK oleh training pipeline (src/data_pipeline)
maupun real-time pipeline (src/gui, src/acquisition) — tidak ada duplikasi
logic filter di tempat lain. apply_full_preprocessing() adalah titik masuk
tunggal yang dipanggil kedua jalur tsb (notch lalu bandpass).
"""

import numpy as np
from scipy.signal import cheby2, filtfilt, iirnotch

from src.config import (
    FS_TARGET,
    BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ,
    BANDPASS_ORDER,
    BANDPASS_RS_DB,
    NOTCH_FREQ_HZ,
    NOTCH_QUALITY_FACTOR,
)


def _filtfilt_safe(b, a, signal: np.ndarray) -> np.ndarray:
    """filtfilt dengan padlen yang otomatis diperkecil untuk sinyal pendek,
    supaya epoch/sinyal singkat (mis. di unit test) tidak crash."""
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(signal) <= padlen:
        padlen = max(0, len(signal) - 1)
    return filtfilt(b, a, signal, padlen=padlen if padlen > 0 else None)


def apply_notch_filter(
    signal: np.ndarray,
    fs: int = FS_TARGET,
    freq: float = NOTCH_FREQ_HZ,
    quality_factor: float = NOTCH_QUALITY_FACTOR,
) -> np.ndarray:
    """Notch filter IIR (scipy.signal.iirnotch) untuk membuang interferensi
    jala-jala listrik (powerline interference) di freq Hz (default 50 Hz).

    Q (quality_factor) tinggi -> notch sempit di sekitar freq, meminimalkan
    distorsi pada komponen ECG di frekuensi lain. filtfilt dipakai (bukan
    lfilter) untuk konsistensi zero-phase dengan apply_bandpass_cheby2.

    Diterapkan SEBELUM bandpass di apply_full_preprocessing(): notch
    membuang interferensi narrowband yang dominan terlebih dahulu, baru
    bandpass membentuk ulang spektrum keseluruhan sinyal.
    """
    signal = np.asarray(signal, dtype=float)
    nyquist = fs / 2.0
    b, a = iirnotch(freq / nyquist, quality_factor)
    return _filtfilt_safe(b, a, signal)


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
    return _filtfilt_safe(b, a, signal)


def apply_full_preprocessing(signal: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """Titik masuk tunggal preprocessing ECG: notch 50Hz -> bandpass 2-40Hz.

    Dipakai identik oleh src/data_pipeline/build_features.py (training) dan
    src/gui/app.py (real-time) supaya tidak ada duplikasi/divergensi logic
    filter antar kedua jalur.
    """
    notched = apply_notch_filter(signal, fs=fs)
    return apply_bandpass_cheby2(notched, fs=fs)
