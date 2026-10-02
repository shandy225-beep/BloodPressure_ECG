"""
Epoching sinyal ECG (10 detik = EPOCH_LEN sampel) + ekstraksi target SBP/DBP
dari sinyal ABP menggunakan peakutils (konsisten dengan workflow lama).

# Bagian 3.x proposal — Segmentasi epoch & anotasi target tekanan darah

Training memakai segment_epochs()/build_epoch_dataset() pada rekaman yang sudah
difilter utuh. Real-time memakai FilteredEpochBuffer (buffer sampel stream yang
sudah di-resample ke FS_TARGET, difilter dengan konteks kiri/kanan) supaya epoch
yang dihasilkan setara dengan training.
"""

import logging
from typing import Iterator, Optional, Tuple

import numpy as np
import peakutils

from src.config import FS_TARGET, EPOCH_LEN, FILTER_CONTEXT_LEN
from src.preprocessing.filters import apply_full_preprocessing

logger = logging.getLogger(__name__)

# Non-overlapping dipilih sebagai default: dataset (12 part, ~12 ribu rekaman x
# ~60000 sampel) sudah menghasilkan ~260 ribu epoch valid sehingga tidak perlu
# overlap untuk memperbanyak data latih. Overlap bisa diaktifkan lewat parameter
# `hop_len` bila data latih di masa depan terbatas.


def segment_epochs(signal: np.ndarray, epoch_len: int = EPOCH_LEN, hop_len: Optional[int] = None) -> Iterator[np.ndarray]:
    """Segmentasi sinyal 1D menjadi jendela epoch_len sampel.

    hop_len=None -> non-overlapping (hop_len=epoch_len).
    """
    hop = hop_len if hop_len is not None else epoch_len
    n = len(signal)
    for start in range(0, n - epoch_len + 1, hop):
        yield signal[start : start + epoch_len]


def extract_sbp_dbp_from_abp(abp_epoch: np.ndarray, fs: int = FS_TARGET) -> Optional[Tuple[float, float]]:
    """Ekstrak SBP (puncak maksimum) & DBP (lembah sebelum siklus berikutnya)
    dari satu epoch sinyal ABP, memakai peakutils.indexes.

    SBP = rata-rata amplitudo puncak sistolik dalam epoch.
    DBP = rata-rata amplitudo lembah diastolik (di antara dua puncak sistolik
    berurutan) dalam epoch.
    """
    abp_epoch = np.asarray(abp_epoch, dtype=float)
    min_dist = max(1, int(0.25 * fs))  # jarak minimum antar detak jantung (~240 bpm max)

    try:
        sys_locs = peakutils.indexes(abp_epoch, thres=0.3, min_dist=min_dist)
    except Exception as e:
        logger.warning("Deteksi puncak sistolik gagal: %s", e)
        return None

    if len(sys_locs) < 2:
        return None

    dia_locs = []
    for i in range(len(sys_locs) - 1):
        seg_start, seg_end = sys_locs[i], sys_locs[i + 1]
        local_min_idx = np.argmin(abp_epoch[seg_start:seg_end])
        dia_locs.append(seg_start + local_min_idx)

    if not dia_locs:
        return None

    sbp = float(np.mean(abp_epoch[sys_locs]))
    dbp = float(np.mean(abp_epoch[dia_locs]))

    if not (np.isfinite(sbp) and np.isfinite(dbp)) or sbp <= dbp:
        return None

    return sbp, dbp


def build_epoch_dataset(ecg: np.ndarray, abp: np.ndarray, fs: int = FS_TARGET, epoch_len: int = EPOCH_LEN):
    """Pasangkan setiap epoch ECG dengan target SBP/DBP dari epoch ABP yang bersesuaian."""
    for ecg_epoch, abp_epoch in zip(segment_epochs(ecg, epoch_len), segment_epochs(abp, epoch_len)):
        target = extract_sbp_dbp_from_abp(abp_epoch, fs)
        if target is None:
            continue
        sbp, dbp = target
        yield ecg_epoch, sbp, dbp


class FilteredEpochBuffer:
    """Buffer real-time yang menghasilkan epoch SUDAH TERFILTER dan setara
    dengan pipeline training.

    Training memfilter seluruh rekaman lalu memotong epoch, sehingga tepi
    epoch tidak terkena efek tepi filtfilt. Memfilter tiap epoch 10 dtk
    secara terpisah menimbulkan efek tepi di kedua ujung epoch -> fitur
    (mean, CV, median, ...) bergeser jauh dari distribusi training. Di sini
    jendela yang difilter diperlebar `context_len` sampel di kiri & kanan
    epoch, lalu hanya bagian tengahnya (epoch) yang diambil.

    Konsekuensi: epoch k baru keluar setelah `context_len` sampel SETELAH
    akhir epoch tsb diterima (tertunda context_len/fs detik). Epoch pertama
    tidak punya konteks kiri -- identik dengan epoch pertama pada training
    (awal rekaman juga tanpa konteks kiri). Sampel harus sudah di-resample
    ke fs pipeline (FS_TARGET) sebelum masuk sini.
    """

    def __init__(self, epoch_len: int = EPOCH_LEN, context_len: int = FILTER_CONTEXT_LEN, fs: int = FS_TARGET):
        self.epoch_len = epoch_len
        self.context_len = context_len
        self.fs = fs
        self.reset()

    def reset(self):
        self._buffer = []           # sampel mentah, mulai dari indeks global _buffer_start
        self._buffer_start = 0
        self._n_total = 0           # total sampel yang sudah diterima
        self._next_epoch_start = 0  # indeks global awal epoch berikutnya

    def add_sample(self, value: float) -> Optional[np.ndarray]:
        """Tambah satu sampel mentah; kembalikan epoch terfilter bila sudah siap."""
        self._buffer.append(value)
        self._n_total += 1

        epoch_start = self._next_epoch_start
        epoch_end = epoch_start + self.epoch_len
        window_end = epoch_end + self.context_len
        if self._n_total < window_end:
            return None

        window_start = max(0, epoch_start - self.context_len)
        offset = self._buffer_start
        window = np.asarray(self._buffer[window_start - offset : window_end - offset], dtype=float)
        filtered = apply_full_preprocessing(window, fs=self.fs)

        left = epoch_start - window_start
        epoch = filtered[left : left + self.epoch_len]

        # Buang sampel yang tak diperlukan lagi: epoch berikutnya butuh
        # context_len sampel sebelum awalnya (= akhir epoch ini).
        self._next_epoch_start = epoch_end
        keep_from = max(offset, epoch_end - self.context_len)
        del self._buffer[: keep_from - offset]
        self._buffer_start = keep_from
        return epoch

    def __len__(self):
        return len(self._buffer)
