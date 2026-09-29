"""
Aplikasi GUI real-time (PyQt5 + pyqtgraph): akuisisi ECG dari sensor Shimmer3R
-> resample ke FS_TARGET -> buffer epoch 1250 sampel + filter berkonteks
(FilteredEpochBuffer) -> ekstraksi 25 fitur -> prediksi EBT (SBP/DBP) -> tampilkan.

# Bagian 5.x proposal — Aplikasi real-time
"""

import csv
import logging
import sys
from datetime import datetime

import joblib
import numpy as np
import pyqtgraph as pg
import serial.tools.list_ports
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSpacerItem,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.config import (
    FS_TARGET, EPOCH_LEN, GUI_PLOT_WINDOW_SEC, GUI_UPDATE_INTERVAL_MS, GUI_DISPLAY_LAG_SEC,
    FILTER_CONTEXT_SEC, NOTCH_FREQ_HZ, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ,
    SBP_MODEL_PATH, DBP_MODEL_PATH, FEATURE_NAMES,
)
from src.preprocessing.filters import apply_full_preprocessing
from src.features.time_domain import extract_features
from src.data_pipeline.epoching import FilteredEpochBuffer
from src.acquisition.shimmer_interface import ShimmerECGStream

logger = logging.getLogger(__name__)


class Predictor:
    """Muat model EBT + validasi metadata FS_TARGET/EPOCH_LEN sebelum prediksi."""

    def __init__(self, sbp_path: str = SBP_MODEL_PATH, dbp_path: str = DBP_MODEL_PATH):
        self.model_sbp = None
        self.model_dbp = None
        self.ok = False
        try:
            meta_sbp = joblib.load(sbp_path)
            meta_dbp = joblib.load(dbp_path)
            self._validate_metadata(meta_sbp, "SBP")
            self._validate_metadata(meta_dbp, "DBP")
            self.model_sbp = meta_sbp["model"]
            self.model_dbp = meta_dbp["model"]
            self.ok = True
            logger.info("Model EBT SBP & DBP berhasil dimuat.")
        except FileNotFoundError:
            logger.warning("Model belum tersedia di %s / %s — jalankan src/model/train.py terlebih dahulu.", sbp_path, dbp_path)
        except Exception as e:
            logger.error("Gagal memuat model: %s", e)

    @staticmethod
    def _validate_metadata(meta: dict, label: str):
        fs = meta.get("fs")
        epoch_len = meta.get("epoch_len")
        if fs != FS_TARGET or epoch_len != EPOCH_LEN:
            raise ValueError(
                f"Metadata model {label} tidak cocok (fs={fs}, epoch_len={epoch_len}) "
                f"dengan konfigurasi aktif (FS_TARGET={FS_TARGET}, EPOCH_LEN={EPOCH_LEN})."
            )

    def predict(self, epoch: np.ndarray):
        if not self.ok:
            return None, None
        feats = extract_features(epoch)
        X = np.array([[feats[name] for name in FEATURE_NAMES]])
        sbp = float(self.model_sbp.predict(X)[0])
        dbp = float(self.model_dbp.predict(X)[0])
        return sbp, dbp


class AcquisitionWorker(QThread):
    """Thread pembaca stream Shimmer -> emit sampel mentah."""

    new_sample = pyqtSignal(float)
    status_changed = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self._running = True

    def run(self):
        try:
            self.stream.connect()
            self.status_changed.emit(f"Connected (native_fs={self.stream.native_fs:.1f} Hz)")
            self.stream.start_stream()
            while self._running:
                sample = self.stream.read_sample(timeout=0.5)
                if sample is not None:
                    self.new_sample.emit(sample)
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            try:
                self.stream.stop_stream()
            except Exception:
                pass

    def stop(self):
        self._running = False


class BPMonitorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BP-ECG Monitor — Estimasi Tekanan Darah dari Sinyal ECG (EBT)")
        self.resize(1200, 700)
        self.setStyleSheet("""
            QWidget { background-color: #fafafa; font-family: Segoe UI; color: #222; }
            QLabel { font-size: 14px; }
            QPushButton {
                background-color: #1976D2; color: white; border-radius: 6px;
                padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1565C0; }
            QComboBox { padding: 4px; background-color: white; border: 1px solid #ccc; border-radius: 4px; }
        """)

        self.predictor = Predictor()
        # Buffer ini yang memfilter (notch+bandpass) dengan konteks kiri/kanan,
        # supaya epoch setara dengan pipeline training (lihat FilteredEpochBuffer).
        self.epoch_buffer = FilteredEpochBuffer(epoch_len=EPOCH_LEN, fs=FS_TARGET)
        self.history = []  # list of (timestamp, sbp, dbp)

        # Plot: buffer sampel MENTAH (hasil resample) + hitungan sampel total.
        # Filtering dilakukan saat menggambar (lihat _update_plot) dengan
        # apply_full_preprocessing yang sama dengan model. Waktu tiap sampel =
        # nomor sampel / FS_TARGET (bukan time.time(): sampel tiba berkelompok
        # dari resampler, sehingga jam dinding membuat kurva bergerigi).
        self.plot_buffer = []
        self.n_samples_total = 0
        self.plot_dirty = False
        self.display_lag = int(round(GUI_DISPLAY_LAG_SEC * FS_TARGET))
        # cukup untuk: jendela tampil + ujung kanan yang ditahan + konteks kiri
        # supaya efek tepi filter di ujung kiri buffer tidak terlihat.
        self.max_plot_samples = int(FS_TARGET * (GUI_PLOT_WINDOW_SEC + GUI_DISPLAY_LAG_SEC + FILTER_CONTEXT_SEC))

        self.worker = None

        self._build_ui()

        self.timer = QTimer()
        self.timer.setInterval(GUI_UPDATE_INTERVAL_MS)
        self.timer.timeout.connect(self._update_plot)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        title = QLabel("Sistem Pemantauan Tekanan Darah Berbasis Sinyal ECG (Ensemble Bagged Tree)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 15pt; font-weight: bold; color: #1976D2;")
        main_layout.addWidget(title)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)

        # ----- Panel kiri: plot + kontrol -----
        left_panel = QVBoxLayout()

        self.pg_plot = pg.PlotWidget(title=(
            f"Sinyal ECG Real-time (resample {FS_TARGET} Hz, notch {NOTCH_FREQ_HZ:g} Hz + "
            f"bandpass {BANDPASS_LOW_HZ:g}-{BANDPASS_HIGH_HZ:g} Hz; tampilan tertunda {GUI_DISPLAY_LAG_SEC:g} dtk)"
        ))
        self.pg_plot.setBackground('w')
        self.pg_plot.showGrid(x=True, y=True, alpha=0.1)
        self.pg_plot.setLabel('left', 'Amplitudo (mV)')
        self.pg_plot.setLabel('bottom', 'Waktu (s)')
        self.ecg_curve = self.pg_plot.plot(pen=pg.mkPen(color='#1976D2', width=2))
        left_panel.addWidget(self.pg_plot, stretch=1)

        # status baris
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Not Connected")
        self.status_label.setStyleSheet("font-size:13px; font-weight:bold; color:#b71c1c;")
        self.native_fs_label = QLabel("native_fs: -")
        self.resample_label = QLabel(f"resample -> FS_TARGET={FS_TARGET}Hz: -")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.native_fs_label)
        status_layout.addWidget(self.resample_label)
        left_panel.addLayout(status_layout)

        # kontrol koneksi Shimmer3R
        control_layout = QHBoxLayout()
        self.port_combo = QComboBox()
        self._refresh_ports()

        self.refresh_btn = QPushButton("Refresh Port")
        self.refresh_btn.clicked.connect(self._refresh_ports)

        self.start_btn = QPushButton("Mulai Pengukuran")
        self.start_btn.clicked.connect(self._start_measurement)
        self.stop_btn = QPushButton("Berhenti")
        self.stop_btn.clicked.connect(self._stop_measurement)
        self.stop_btn.setEnabled(False)

        control_layout.addWidget(QLabel("Port Shimmer3R:"))
        control_layout.addWidget(self.port_combo)
        control_layout.addWidget(self.refresh_btn)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        left_panel.addLayout(control_layout)

        # tabel histori
        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["Timestamp", "SBP (mmHg)", "DBP (mmHg)"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        left_panel.addWidget(self.history_table, stretch=1)

        export_btn = QPushButton("Ekspor CSV")
        export_btn.clicked.connect(self._export_csv)
        left_panel.addWidget(export_btn)

        content_layout.addLayout(left_panel, 3)

        # ----- Panel kanan: kartu BP -----
        right_panel = QVBoxLayout()
        right_panel.setSpacing(20)

        bp_frame = QFrame()
        bp_frame.setStyleSheet("QFrame { background-color: #fff; border-radius: 12px; border: 1px solid #ddd; padding: 12px; }")
        bp_layout = QVBoxLayout(bp_frame)
        bp_title = QLabel("Estimasi Tekanan Darah")
        bp_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #1976D2;")
        self.bp_value_label = QLabel("--/--")
        self.bp_value_label.setAlignment(Qt.AlignCenter)
        self.bp_value_label.setStyleSheet("font-size: 36pt; font-weight: bold; color: #333;")
        bp_unit = QLabel("mmHg (SBP/DBP)")
        bp_unit.setAlignment(Qt.AlignCenter)
        bp_unit.setStyleSheet("font-size: 10pt; color: gray;")
        bp_layout.addWidget(bp_title)
        bp_layout.addWidget(self.bp_value_label)
        bp_layout.addWidget(bp_unit)
        right_panel.addWidget(bp_frame)

        model_frame = QFrame()
        model_frame.setStyleSheet("QFrame { background-color: #fff; border-radius: 12px; border: 1px solid #ddd; padding: 12px; }")
        model_layout = QVBoxLayout(model_frame)
        model_status = "Model dimuat" if self.predictor.ok else "Model TIDAK ditemukan — jalankan src/model/train.py"
        model_color = "#1b5e20" if self.predictor.ok else "#b71c1c"
        self.model_status_label = QLabel(model_status)
        self.model_status_label.setStyleSheet(f"font-size: 10pt; color: {model_color};")
        self.model_status_label.setWordWrap(True)
        model_layout.addWidget(self.model_status_label)
        right_panel.addWidget(model_frame)

        right_panel.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        content_layout.addLayout(right_panel, 1)

    def _refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.port_combo.addItem("Tidak ada port terdeteksi")
        else:
            for p in ports:
                self.port_combo.addItem(p.device)

    def _build_stream(self):
        port = self.port_combo.currentText()
        if "Tidak ada" in port:
            raise RuntimeError("Tidak ada port serial yang bisa digunakan untuk Shimmer.")
        return ShimmerECGStream(port=port)

    def _start_measurement(self):
        try:
            stream = self._build_stream()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self.plot_buffer = []
        self.n_samples_total = 0
        self.plot_dirty = False
        self.epoch_buffer.reset()
        self.status_label.setText("Connecting...")
        self.status_label.setStyleSheet("font-size:13px; font-weight:bold; color:#f9a825;")

        self.worker = AcquisitionWorker(stream)
        self.worker.new_sample.connect(self._on_new_sample)
        self.worker.status_changed.connect(self._on_status_changed)
        self.worker.error_signal.connect(self._on_stream_error)
        self.worker.start()
        self.timer.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop_measurement(self):
        self.timer.stop()
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

        self.status_label.setText("Not Connected")
        self.status_label.setStyleSheet("font-size:13px; font-weight:bold; color:#b71c1c;")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_status_changed(self, text):
        self.status_label.setText(text)
        self.status_label.setStyleSheet("font-size:13px; font-weight:bold; color:#1b5e20;")
        native_fs = getattr(self.worker.stream, "native_fs", None)
        if native_fs is not None:
            self.native_fs_label.setText(f"native_fs: {native_fs:.1f} Hz")
            active = "aktif" if native_fs != FS_TARGET else "identity (native==target)"
            self.resample_label.setText(f"resample -> FS_TARGET={FS_TARGET}Hz: {active}")

    def _on_stream_error(self, msg):
        QMessageBox.warning(self, "Stream Error", msg)
        self._stop_measurement()

    def _on_new_sample(self, value: float):
        self.plot_buffer.append(value)
        self.n_samples_total += 1
        self.plot_dirty = True
        if len(self.plot_buffer) > self.max_plot_samples:
            excess = len(self.plot_buffer) - self.max_plot_samples
            self.plot_buffer = self.plot_buffer[excess:]

        completed_epoch = self.epoch_buffer.add_sample(value)
        if completed_epoch is not None:
            self._process_epoch(completed_epoch)

    def _process_epoch(self, epoch: np.ndarray):
        # epoch sudah terfilter oleh FilteredEpochBuffer -- JANGAN difilter lagi.
        sbp, dbp = self.predictor.predict(epoch)
        if sbp is None:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append((timestamp, sbp, dbp))

        self.bp_value_label.setText(f"{int(round(sbp))}/{int(round(dbp))}")

        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        self.history_table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.history_table.setItem(row, 1, QTableWidgetItem(f"{sbp:.1f}"))
        self.history_table.setItem(row, 2, QTableWidgetItem(f"{dbp:.1f}"))
        self.history_table.scrollToBottom()

    @pyqtSlot()
    def _update_plot(self):
        # Belum ada sampel baru -> tak perlu filter ulang.
        if not self.plot_dirty or len(self.plot_buffer) <= self.display_lag:
            return
        self.plot_dirty = False

        raw = np.array(self.plot_buffer)
        # Filter yang SAMA dengan yang dilihat model (zero-phase). Ujung kanan
        # (display_lag sampel terakhir) dibuang karena masih kena efek tepi
        # filtfilt; ujung kiri buffer sudah di luar jendela tampil.
        filtered = apply_full_preprocessing(raw, fs=FS_TARGET)[: -self.display_lag]
        first_index = self.n_samples_total - len(raw)
        tdata = (first_index + np.arange(len(filtered))) / FS_TARGET
        self.ecg_curve.setData(tdata, filtered)

        t_max = tdata[-1]
        t_min = max(0, t_max - GUI_PLOT_WINDOW_SEC)
        self.pg_plot.setXRange(t_min, t_max)

    def _export_csv(self):
        if not self.history:
            QMessageBox.information(self, "Ekspor CSV", "Belum ada data pengukuran untuk diekspor.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Simpan CSV", "bp_history.csv", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "SBP", "DBP"])
            writer.writerows(self.history)
        QMessageBox.information(self, "Ekspor CSV", f"Data tersimpan ke {path}")

    def closeEvent(self, event):
        self._stop_measurement()
        event.accept()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = QApplication(sys.argv)
    win = BPMonitorGUI()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
