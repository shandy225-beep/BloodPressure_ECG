"""
ShimmerECGStream — interface koneksi Shimmer3R ECG via pyserial + pyshimmer.

Logic koneksi/parsing paket di-REUSE dari kode lama project ini
(main_app_fix.py, kelas ShimmerReader): pembukaan serial.Serial ->
ShimmerBluetooth.initialize() -> add_stream_callback() -> start_streaming(),
serta konversi ADC 24-bit ke milivolt (adc_to_millivolts) memakai channel
EChannelType.EXG_ADS1292R_2_CH1_24BIT.

Perbedaan terhadap kode lama: kelas lama (ShimmerReader) adalah QThread yang
langsung emit ke GUI PyQt. Di sini logic koneksi dibungkus ke interface
generik (connect/start_stream/read_sample/stop_stream/native_fs), dan
native_fs DIBACA dari device lewat get_sampling_rate() (bukan diasumsikan)
lalu setiap sampel di-resample ke FS_TARGET sebelum masuk queue (lihat
resample_to_target()).
"""

import logging
import threading
import time
from queue import Empty, Queue
from typing import Optional

import serial

from src.config import FS_TARGET
from src.acquisition.resampler import resample_to_target

logger = logging.getLogger(__name__)

try:
    from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
    SHIMMER_AVAILABLE = True
except ImportError:
    SHIMMER_AVAILABLE = False
    DEFAULT_BAUDRATE = 115200


def adc_to_millivolts(adc_signal: float, gain: float = 6, offset: float = 0, vref: float = 2.42, adc_bits: int = 24) -> float:
    """Konversi ADC 24-bit Shimmer EXG ke milivolt. Diambil apa adanya dari
    main_app_fix.py (ShimmerReader.adc_to_milivotls)."""
    adc_sensitivity = (vref * 1000) / (2 ** (adc_bits - 1) - 1)
    return ((adc_signal - offset) * adc_sensitivity) / gain


class ShimmerECGStream:
    """Interface real-time Shimmer3R ECG. Sample keluaran sudah di-resample
    ke FS_TARGET (lihat _emit_raw_sample) sebelum tersedia lewat read_sample().
    """

    def __init__(self, port: str, baudrate: Optional[int] = None):
        if not SHIMMER_AVAILABLE:
            raise ImportError("pyshimmer tidak terpasang; tidak bisa memakai ShimmerECGStream.")

        self.port = port
        self.baudrate = baudrate or DEFAULT_BAUDRATE

        self.native_fs: Optional[float] = None
        self._shim_dev: Optional["ShimmerBluetooth"] = None
        self._serial_conn: Optional[serial.Serial] = None

        self._queue: "Queue[float]" = Queue()
        self._raw_buffer = []          # buffer sampel native_fs sebelum resample
        self._resample_chunk_sec = 1.0  # resample per potongan 1 detik (latensi kecil, ratio stabil)

        self._running = False
        self._connected = False

    def connect(self):
        self._serial_conn = serial.Serial(self.port, self.baudrate, rtscts=False, dsrdtr=False)
        self._shim_dev = ShimmerBluetooth(self._serial_conn)
        self._shim_dev.initialize()

        dev_name = self._shim_dev.get_device_name()
        self.native_fs = float(self._shim_dev.get_sampling_rate())
        logger.info("Terhubung ke Shimmer '%s', native_fs=%.2f Hz", dev_name, self.native_fs)

        self._shim_dev.add_stream_callback(self._handle_packet)
        self._connected = True
        return True

    def _handle_packet(self, pkt: "DataPacket"):
        try:
            if EChannelType.EXG_ADS1292R_2_CH1_24BIT in pkt._values:
                ecg_raw = pkt[EChannelType.EXG_ADS1292R_2_CH1_24BIT]
                ecg_mv = adc_to_millivolts(ecg_raw, gain=6, offset=0)
                self._raw_buffer.append(ecg_mv)

                chunk_size = max(1, int(self._resample_chunk_sec * self.native_fs))
                if len(self._raw_buffer) >= chunk_size:
                    self._flush_resampled_chunk()
        except Exception as e:
            logger.error("Error handling Shimmer packet: %s", e)

    def _flush_resampled_chunk(self):
        chunk = self._raw_buffer
        self._raw_buffer = []
        resampled = resample_to_target(chunk, native_fs=self.native_fs, target_fs=FS_TARGET)
        for value in resampled:
            self._queue.put(float(value))

    def start_stream(self):
        if not self._connected:
            raise RuntimeError("Panggil connect() sebelum start_stream()")
        self._shim_dev.start_streaming()
        self._running = True

    def read_sample(self, timeout: float = 0.5) -> Optional[float]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def stop_stream(self):
        self._running = False
        if self._raw_buffer:
            self._flush_resampled_chunk()

        if self._shim_dev is not None:
            try:
                self._shim_dev.stop_streaming()
            except Exception as e:
                logger.warning("stop_streaming error: %s", e)
            try:
                self._shim_dev.shutdown()
            except Exception as e:
                logger.warning("shutdown error: %s", e)

        if self._serial_conn is not None:
            try:
                self._serial_conn.close()
            except Exception:
                pass

        self._connected = False
