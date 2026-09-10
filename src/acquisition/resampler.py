"""
Resampling sinyal akuisisi real-time (native_fs sensor Shimmer) ke FS_TARGET.

Wajib dipanggil di titik keluaran stream (lihat shimmer_interface.py) sebelum
data masuk ke buffer epoch, filter, dan ekstraksi fitur — supaya pipeline
real-time identik secara statistik dengan pipeline training (part_1.mat,
native 125Hz).
"""

import logging
import math

import numpy as np
from scipy.signal import resample, resample_poly

from src.config import FS_TARGET

logger = logging.getLogger(__name__)


def resample_to_target(signal: np.ndarray, native_fs: float, target_fs: int = FS_TARGET) -> np.ndarray:
    """Resample signal dari native_fs ke target_fs.

    Preferensi: scipy.signal.resample_poly, karena secara internal memakai
    FIR polyphase filter (lebih stabil dan tidak membutuhkan asumsi
    periodisitas sinyal, tidak seperti scipy.signal.resample yang berbasis
    FFT dan bisa menimbulkan artefak Gibbs pada sinyal non-periodik seperti
    ECG). up/down dihitung dari rasio target_fs/native_fs yang disederhanakan
    lewat GCD supaya faktor interpolasi/desimasi sekecil mungkin (lebih cepat
    & filter FIR lebih pendek).

    Bila native_fs bukan bilangan bulat (device melaporkan fs pecahan, mis.
    511.9 Hz) rasio disederhanakan lewat pembulatan ke resolusi 0.001 Hz
    sebelum dicari GCD-nya; jika hasil resample_poly gagal/aneh untuk rasio
    tersebut, fallback ke scipy.signal.resample (FFT-based) yang menerima fs
    non-integer secara langsung lewat argumen num sample.
    """
    signal = np.asarray(signal, dtype=float)

    if native_fs == target_fs:
        return signal.copy()

    if len(signal) == 0:
        return signal.copy()

    try:
        # Sederhanakan rasio up/down lewat GCD; kalikan 1000x dulu untuk
        # menoleransi native_fs pecahan (mis. 511.9 Hz -> 511900/125000).
        native_fs_milli = int(round(native_fs * 1000))
        target_fs_milli = int(round(target_fs * 1000))
        g = math.gcd(native_fs_milli, target_fs_milli)
        up = target_fs_milli // g
        down = native_fs_milli // g

        resampled = resample_poly(signal, up, down)
        return resampled
    except Exception as e:
        logger.warning(
            "resample_poly gagal untuk native_fs=%s -> target_fs=%s (%s), fallback ke scipy.signal.resample",
            native_fs, target_fs, e,
        )
        n_target = max(1, int(round(len(signal) * target_fs / native_fs)))
        return resample(signal, n_target)
