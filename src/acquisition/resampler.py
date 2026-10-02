"""
Resampling sinyal akuisisi (native_fs sensor Shimmer) ke FS_TARGET, supaya
pipeline real-time identik secara statistik dengan pipeline training (dataset
native 125Hz).

- StreamingResampler: untuk STREAM real-time; dipanggil di titik keluaran stream
  (shimmer_interface.py) sebelum data masuk ke buffer epoch, filter, dan fitur.
- resample_to_target(): untuk sinyal utuh/offline. JANGAN dipanggil berulang pada
  potongan pendek stream -- tiap batas potongan menjadi artefak (lihat
  docstring StreamingResampler).
"""

import logging
import math

import numpy as np
from scipy.signal import firwin, resample, resample_poly

from src.config import FS_TARGET

logger = logging.getLogger(__name__)


def _simplified_ratio(native_fs: float, target_fs: float):
    """(up, down) terkecil untuk target_fs/native_fs; native_fs pecahan
    ditoleransi lewat resolusi 0.001 Hz (mis. 511.9 Hz -> 511900/125000)."""
    native_milli = int(round(native_fs * 1000))
    target_milli = int(round(target_fs * 1000))
    g = math.gcd(native_milli, target_milli)
    return target_milli // g, native_milli // g


def resample_to_target(signal: np.ndarray, native_fs: float, target_fs: int = FS_TARGET) -> np.ndarray:
    """Resample sinyal UTUH (offline) dari native_fs ke target_fs. Untuk stream
    real-time pakai StreamingResampler.

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
        up, down = _simplified_ratio(native_fs, target_fs)

        resampled = resample_poly(signal, up, down)
        return resampled
    except Exception as e:
        logger.warning(
            "resample_poly gagal untuk native_fs=%s -> target_fs=%s (%s), fallback ke scipy.signal.resample",
            native_fs, target_fs, e,
        )
        n_target = max(1, int(round(len(signal) * target_fs / native_fs)))
        return resample(signal, n_target)


class StreamingResampler:
    """Resampler polyphase STATEFUL untuk stream real-time: native_fs -> target_fs.

    Memanggil resample_to_target() per potongan pendek (mis. 1 detik) salah
    untuk stream: resample_poly mengisi tepi potongan dengan nol,
    sehingga tiap batas potongan menjadi loncatan buatan (terlebih pada
    sinyal ber-DC-offset seperti keluaran ADC), dan jumlah sampel keluaran
    dibulatkan ke atas per potongan (drift timing pada native_fs pecahan).

    Kelas ini memakai FIR yang SAMA dengan scipy.signal.resample_poly
    (firwin, jendela Kaiser beta=5, panjang 2*10*max(up,down)+1), tetapi
    menyimpan riwayat sampel antar-panggilan sehingga output identik berapa
    pun ukuran potongan yang disuapkan (bahkan 1 sampel sekali masuk) dan
    identik dengan resample_poly seluruh sinyal di bagian interior.

    Output ke-m berada pada waktu m/target_fs (m=0 sejajar sampel input
    pertama) -- sama dengan resample_poly, tanpa pergeseran fase.

    Tepi: sebelum sampel pertama, sinyal diperpanjang dengan nilai sampel
    pertama (bukan nol) supaya tidak ada transien awal akibat DC offset.
    Di ujung kanan, output tertahan ~half_len/up sampel native (puluhan ms)
    sampai sampel "masa depan" cukup; flush() melepasnya saat stream berhenti.
    """

    _MAX_ROWS = 4096  # batasi ukuran matriks indeks saat satu panggilan membawa banyak sampel

    def __init__(self, native_fs: float, target_fs: int = FS_TARGET):
        self.native_fs = native_fs
        self.target_fs = target_fs
        self.up, self.down = _simplified_ratio(native_fs, target_fs)
        self._identity = self.up == self.down

        if not self._identity:
            max_rate = max(self.up, self.down)
            self._half_len = 10 * max_rate
            h = firwin(2 * self._half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0))
            self._h = h * self.up
            # jumlah tap maksimum yang menyentuh satu sampel keluaran
            self._n_taps = -(-(2 * self._half_len + 1) // self.up)
            self._left_pad = self._half_len // self.up + 1

        self.reset()

    def reset(self):
        self._buf = np.empty(0, dtype=float)
        self._buf_start = 0     # indeks global (native) dari _buf[0]; negatif = padding tepi kiri
        self._m_next = 0        # indeks global output berikutnya
        self._started = False
        self._last_sample = 0.0

    def process(self, samples) -> np.ndarray:
        """Suapkan sampel native (array/list, boleh kosong); kembalikan sampel
        target_fs yang baru siap (bisa kosong)."""
        x = np.asarray(samples, dtype=float).ravel()
        if x.size == 0:
            return np.empty(0, dtype=float)
        self._last_sample = float(x[-1])
        if self._identity:
            return x.copy()

        if not self._started:
            self._buf = np.full(self._left_pad, x[0], dtype=float)
            self._buf_start = -self._left_pad
            self._started = True
        self._buf = np.concatenate([self._buf, x])
        return self._emit()

    def flush(self) -> np.ndarray:
        """Lepaskan sisa output di ujung stream dengan memperpanjang sinyal
        memakai sampel terakhir. Setelah ini resampler sebaiknya di-reset."""
        if self._identity or not self._started:
            return np.empty(0, dtype=float)
        tail = np.full(self._half_len // self.up + 1, self._last_sample, dtype=float)
        self._buf = np.concatenate([self._buf, tail])
        return self._emit()

    def _emit(self) -> np.ndarray:
        buf_end = self._buf_start + len(self._buf)  # indeks global sampel berikutnya yang belum datang
        # output m butuh sampel native sampai floor((m*down + half_len)/up) <= buf_end-1
        m_max = (buf_end * self.up - 1 - self._half_len) // self.down
        if m_max < self._m_next:
            return np.empty(0, dtype=float)

        outputs = []
        for start in range(self._m_next, m_max + 1, self._MAX_ROWS):
            m = np.arange(start, min(start + self._MAX_ROWS, m_max + 1), dtype=np.int64)
            center = m * self.down + self._half_len
            # n terkecil dengan indeks tap h <= 2*half_len: ceil((m*down - half_len)/up)
            n_first = -((self._half_len - m * self.down) // self.up)
            n = n_first[:, None] + np.arange(self._n_taps, dtype=np.int64)[None, :]
            h_idx = center[:, None] - n * self.up
            valid = (h_idx >= 0) & (h_idx <= 2 * self._half_len)
            vals = self._buf[np.clip(n - self._buf_start, 0, len(self._buf) - 1)]
            taps = self._h[np.clip(h_idx, 0, 2 * self._half_len)]
            outputs.append(np.sum(np.where(valid, vals * taps, 0.0), axis=1))

        self._m_next = m_max + 1
        # buang sampel yang tak dibutuhkan lagi oleh output berikutnya
        keep_from = -((self._half_len - self._m_next * self.down) // self.up)
        drop = max(0, min(keep_from - self._buf_start, len(self._buf)))
        self._buf = self._buf[drop:]
        self._buf_start += drop
        return np.concatenate(outputs)
