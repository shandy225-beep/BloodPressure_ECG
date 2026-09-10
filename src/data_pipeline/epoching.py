"""
Epoching sinyal ECG (10 detik = EPOCH_LEN sampel) + ekstraksi target SBP/DBP
dari sinyal ABP menggunakan peakutils (konsisten dengan workflow lama).

# Bagian 3.x proposal — Segmentasi epoch & anotasi target tekanan darah

Fungsi segment_epochs() dipakai baik oleh training pipeline (data historis,
non-overlapping) maupun real-time pipeline lewat kelas EpochBuffer (buffer
1250 sampel dari stream yang sudah di-resample ke FS_TARGET).
"""

import logging
from typing import Iterator, Optional, Tuple

import numpy as np
import peakutils

from src.config import FS_TARGET, EPOCH_LEN

logger = logging.getLogger(__name__)

# Non-overlapping dipilih sebagai default: dataset sekunder (part_1.mat, 1000
# rekaman x ~60000 sampel) sudah cukup besar (~48 epoch/rekaman) sehingga
# tidak perlu overlap untuk memperbanyak data latih. Overlap bisa diaktifkan
# lewat parameter `hop_len` bila data latih di masa depan terbatas.


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


class EpochBuffer:
    """Buffer akumulasi sampel untuk mode real-time.

    Dipakai oleh acquisition/shimmer_interface.py setelah sinyal di-resample
    ke FS_TARGET: setiap sampel ditambahkan lewat add_sample(), begitu buffer
    mencapai EPOCH_LEN sampel, add_sample() mengembalikan epoch penuh dan
    mereset buffer (non-overlapping, sama seperti training).
    """

    def __init__(self, epoch_len: int = EPOCH_LEN):
        self.epoch_len = epoch_len
        self._buffer = []

    def add_sample(self, value: float) -> Optional[np.ndarray]:
        self._buffer.append(value)
        if len(self._buffer) >= self.epoch_len:
            epoch = np.array(self._buffer[: self.epoch_len])
            self._buffer = self._buffer[self.epoch_len :]
            return epoch
        return None

    def reset(self):
        self._buffer = []

    def __len__(self):
        return len(self._buffer)
