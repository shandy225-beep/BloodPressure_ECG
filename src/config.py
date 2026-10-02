"""
Konfigurasi global sistem BP-ECG Monitor.

Aturan penting: FS_TARGET didefinisikan HANYA di sini.
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

# Konteks (detik) di kiri & kanan epoch saat filtering real-time. filtfilt
# zero-phase punya efek tepi di ujung jendela; training memfilter SELURUH
# rekaman lalu memotong epoch, jadi real-time harus memfilter jendela yang
# lebih lebar dari epoch lalu memotong bagian tengahnya. Diukur pada rekaman
# asli: margin 3 dtk -> selisih sinyal vs filter-seluruh-rekaman ~1e-9
# (0 dtk: ~1.0, 2 dtk: ~1e-7). Harga yang dibayar: hasil tiap epoch tertunda
# FILTER_CONTEXT_SEC detik (butuh sampel "masa depan" untuk sisi kanan).
FILTER_CONTEXT_SEC = 3
FILTER_CONTEXT_LEN = FILTER_CONTEXT_SEC * FS_TARGET   # 375 sampel

# ---------------------------------------------------------------------------
# Filter bandpass ECG (Persamaan 2.x — preprocessing sinyal ECG)
# ---------------------------------------------------------------------------
BANDPASS_LOW_HZ = 2.0
BANDPASS_HIGH_HZ = 40.0
BANDPASS_ORDER = 4
BANDPASS_RS_DB = 40       # stopband attenuation (dB) untuk Chebyshev Type II

# ---------------------------------------------------------------------------
# Filter notch (powerline interference)
# ---------------------------------------------------------------------------
# 50 Hz dipakai sebagai default (standar jaringan listrik Indonesia/Eropa).
# Ganti ke 60.0 kalau akuisisi dilakukan di jaringan 60Hz (mis. Amerika).
NOTCH_FREQ_HZ = 50.0
NOTCH_QUALITY_FACTOR = 30.0   # Q tinggi -> notch sempit, minim distorsi di luar 50Hz

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

# Seluruh part_N.mat (part_1 .. part_12) di RAW_DATA_DIR; ditemukan otomatis oleh
# discover_dataset_files() di src/data_pipeline/build_features.py.
DATASET_GLOB_PATTERN = os.path.join(RAW_DATA_DIR, "part_*.mat")

SBP_MODEL_PATH = os.path.join(MODELS_DIR, "ebt_sbp.pkl")
DBP_MODEL_PATH = os.path.join(MODELS_DIR, "ebt_dbp.pkl")

FEATURE_REPORT_PATH = os.path.join(REPORTS_DIR, "eval_report.csv")

# --tag eksperimen: dipakai bersama oleh src/model/train.py (simpan model/laporan
# ke file terpisah, tidak menimpa produksi) dan src/gui/app.py (muat model
# eksperimen tertentu untuk diuji, tanpa mengubah file ini). Logic-nya taruh di
# sini, satu tempat, supaya kedua sisi (simpan & muat) selalu sepakat soal nama
# file yang dihasilkan dari tag yang sama.
_INVALID_TAG_CHARS = set('\\/:*?"<>|')


def sanitize_tag(tag: str) -> str:
    """Validasi --tag: spasi diganti '_' (kenyamanan), karakter path/terlarang
    DITOLAK (supaya tidak sengaja menulis/membaca dari folder lain atau nama
    file rusak)."""
    tag = tag.strip().replace(" ", "_")
    if not tag or any(c in _INVALID_TAG_CHARS for c in tag):
        raise ValueError(
            f"--tag tidak valid: {tag!r}. Hindari kosong dan karakter "
            f"path/terlarang ({''.join(sorted(_INVALID_TAG_CHARS))})."
        )
    return tag


def tagged_path(path: str, tag: str) -> str:
    """Sisipkan tag sebelum ekstensi: models/ebt_sbp.pkl + tag "leaf8" ->
    models/ebt_sbp_leaf8.pkl. tag HARUS sudah lolos sanitize_tag()."""
    root, ext = os.path.splitext(path)
    return f"{root}_{tag}{ext}"


# Riwayat SEMUA eksperimen training (beda dengan FEATURE_REPORT_PATH di atas,
# yang cuma menyimpan hasil model TERAKHIR): setiap kali train_and_save()
# dipanggil, satu baris ditambahkan (append, tidak pernah ditimpa) lewat
# src/model/evaluate.py::log_experiment() -- supaya beberapa percobaan
# PARAM_GRID bisa dibandingkan berdampingan. Lihat README bagian "Mencari
# parameter terbaik".
EXPERIMENT_LOG_PATH = os.path.join(REPORTS_DIR, "experiment_log.csv")

# ---------------------------------------------------------------------------
# Real-time GUI
# ---------------------------------------------------------------------------
GUI_PLOT_WINDOW_SEC = 15      # jendela tampilan sinyal ECG live (detik)
GUI_UPDATE_INTERVAL_MS = 33   # ~30 FPS
# Plot live memakai filter yang SAMA dengan model (zero-phase), sehingga butuh
# sedikit sinyal "masa depan" di ujung kanan: kurva tampil tertunda sekian detik.
# Diukur: margin 1 dtk -> selisih vs filter-seluruh-sinyal ~2e-4 (std ECG ~0.2),
# tak terlihat di plot. Filter kausal (tanpa delay) DITOLAK: morfologi QRS berubah
# (puncak R ~35% lebih kecil, muncul undershoot) sehingga menyesatkan.
GUI_DISPLAY_LAG_SEC = 1.0

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
