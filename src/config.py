"""
Konfigurasi global sistem BP-ECG Monitor.

Aturan penting (lihat prompt v2): FS_TARGET didefinisikan HANYA di sini.
Semua modul lain (filter, epoching, ekstraksi fitur, resampler, GUI) WAJIB
mengimpor FS_TARGET dari modul ini, bukan hardcode angka 125.
"""

import os

# ---------------------------------------------------------------------------
# Sampling rate & epoching
# ---------------------------------------------------------------------------
FS_TARGET = 125          # Hz — satu-satunya sumber kebenaran sample rate pipeline
EPOCH_SEC = 10            # panjang epoch dalam detik
EPOCH_LEN = EPOCH_SEC * FS_TARGET   # 1250 sampel/epoch

# ---------------------------------------------------------------------------
# Filter bandpass ECG (Persamaan 2.x — preprocessing sinyal ECG)
# ---------------------------------------------------------------------------
BANDPASS_LOW_HZ = 2.0
BANDPASS_HIGH_HZ = 40.0
BANDPASS_ORDER = 4
BANDPASS_RS_DB = 40       # stopband attenuation (dB) untuk Chebyshev Type II

# ---------------------------------------------------------------------------
# Path proyek
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset mentah (Kaggle Cuff-Less BP Estimation, part_N.mat) sudah tersedia
# di folder Dataset/ pada instalasi ini — bukan data/raw/ seperti konvensi
# awal, jadi dirujuk langsung di sini alih-alih disalin ulang (part_N.mat
# berukuran ratusan MB per file).
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

DEFAULT_DATASET_FILE = os.path.join(RAW_DATA_DIR, "part_1.mat")

SBP_MODEL_PATH = os.path.join(MODELS_DIR, "ebt_sbp.pkl")
DBP_MODEL_PATH = os.path.join(MODELS_DIR, "ebt_dbp.pkl")

FEATURE_REPORT_PATH = os.path.join(REPORTS_DIR, "eval_report.csv")

# ---------------------------------------------------------------------------
# Real-time GUI
# ---------------------------------------------------------------------------
GUI_PLOT_WINDOW_SEC = 15      # jendela tampilan sinyal ECG live (detik)
GUI_UPDATE_INTERVAL_MS = 33   # ~30 FPS

# Urutan 25 fitur time-domain non-fiducial (harus konsisten training <-> inferensi)
FEATURE_NAMES = [
    "kurtosis", "skewness", "iqr", "coefficient_of_variation",
    "geometric_mean", "harmonic_mean", "hjorth_activity", "hjorth_mobility",
    "hjorth_complexity", "maximum", "median", "mean_absolute_deviation",
    "minimum", "central_moment_10", "mean", "average_curve_length",
    "average_energy", "root_mean_square", "standard_error", "standard_deviation",
    "shape_factor", "svd_dominant", "trimmed_mean_25", "trimmed_mean_50",
    "average_teager_energy",
]

for _dir in (DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR):
    os.makedirs(_dir, exist_ok=True)
