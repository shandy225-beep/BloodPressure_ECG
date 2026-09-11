# BP-ECG Monitor

Sistem estimasi **Systolic (SBP)** dan **Diastolic Blood Pressure (DBP)** dari
sinyal ECG single-lead menggunakan **Ensemble Bagged Tree Regressor (EBT)**.

Rujukan metodologis: Kandaz & Uçar (2025), *"Continuous Blood Pressure
Estimation Based on an Electrocardiography Signal Model"*, IEEE Access.

## Aturan sampling rate — WAJIB dipahami

Seluruh pipeline (training maupun real-time) konsisten memakai
**`FS_TARGET = 125 Hz`**, didefinisikan satu kali di [src/config.py](src/config.py).

- Dataset latih (`Dataset/part_1.mat`...`part_12.mat`) sudah native 125 Hz — tidak di-resample.
- Sensor Shimmer3R streaming pada native sample rate device (dibaca lewat
  `get_sampling_rate()`, bukan diasumsikan) yang bisa berbeda dari 125 Hz.
  Setiap sampel real-time **selalu di-resample ke `FS_TARGET`**
  (`src/acquisition/resampler.py`) sebelum masuk ke filter/epoch/fitur —
  supaya statistik fitur epoch training dan real-time apple-to-apple.
- **Jangan pernah hardcode angka 125** di luar `config.py` — semua modul
  mengimpor `FS_TARGET` dari sana.

## Struktur proyek

```
BloodPressure_ECG/
├── Dataset/                         # part_1.mat ... part_12.mat (dataset Kaggle, sudah tersedia)
├── data/
│   └── processed/                   # fitur hasil ekstraksi (features.csv)
├── src/
│   ├── config.py                    # FS_TARGET=125, EPOCH_SEC=10, path
│   ├── preprocessing/filters.py     # notch 50Hz + bandpass Chebyshev Type II
│   ├── features/time_domain.py      # 25 fitur statistik non-fiducial
│   ├── data_pipeline/
│   │   ├── load_dataset.py          # parsing Dataset/part_N.mat (dataset asli)
│   │   ├── epoching.py              # epoch 1250 sampel + target SBP/DBP
│   │   └── build_features.py        # pipeline penuh (semua part_1-12.mat) -> CSV fitur
│   ├── model/
│   │   ├── train.py                 # EBT (BaggingRegressor) + 10-fold CV
│   │   └── evaluate.py              # MAE, RMSE, MAPE, R²
│   ├── acquisition/
│   │   ├── resampler.py             # resample_to_target()
│   │   └── shimmer_interface.py     # koneksi Shimmer3R (reuse main_app_fix.py)
│   └── gui/app.py                   # aplikasi real-time PyQt5 + pyqtgraph
├── models/                          # ebt_sbp.pkl, ebt_dbp.pkl (+ metadata FS)
├── reports/eval_report.csv
├── evenv/                           # virtual environment (Python 3.14.5)
└── tests/
```

Catatan: dataset mentah (`part_N.mat`) dirujuk langsung dari `Dataset/` alih-alih
disalin ke `data/raw/` — file-nya ratusan MB per part, jadi `DEFAULT_DATASET_FILE`
di `config.py` menunjuk langsung ke `Dataset/part_1.mat`.

## Instalasi

Environment `evenv/` di proyek ini sudah berisi seluruh dependensi
`requirements.txt` (Python 3.14.5). Aktifkan sebelum menjalankan apa pun:

```powershell
.\evenv\Scripts\Activate.ps1
pip install pytest   # untuk menjalankan unit test (tidak dipakai aplikasi)
```

## Dataset

Dataset "Cuff-Less Blood Pressure Estimation" (Kaggle
`mkachuee/BloodPressureDataset`) sudah tersedia di `Dataset/part_1.mat`
(beserta part_2–12.mat). Setiap rekaman berbentuk array `(3, n_samples)`:
baris 0 = PPG, baris 1 = ABP, baris 2 = ECG, native 125 Hz.

`Dataset/part_N.mat` **wajib ada** untuk menjalankan training — tidak ada
mode/fallback data sintetis. Jika tidak ditemukan, `load_mat_records()` akan
melempar `FileNotFoundError` dengan instruksi mengunduh dataset dari Kaggle.

## Menjalankan training

```powershell
# 1. Bangun dataset fitur dari SELURUH part_1.mat..part_12.mat yang ada di
#    Dataset/ (auto-discovery + urut numerik). Proses lama (~25-30 menit
#    untuk ~4.6GB data mentah), berjalan lewat notch 50Hz -> bandpass 2-40Hz
#    -> epoch -> 25 fitur untuk tiap file.
python -m src.data_pipeline.build_features --out data/processed/features.csv

# Proses satu file saja (mis. untuk uji cepat), pakai --mat-path:
python -m src.data_pipeline.build_features --mat-path Dataset/part_1.mat --out data/processed/features_part1.csv

# 2. Latih model EBT (BaggingRegressor + 10-fold CV + GridSearchCV ringan).
#    Dengan seluruh 12 part, jumlah baris fitur jauh lebih besar dari
#    sebelumnya (part_1 saja menghasilkan ~25 ribu epoch) — proses training
#    proporsional lebih lama juga.
python -m src.model.train --features-csv data/processed/features.csv
```

Model tersimpan ke `models/ebt_sbp.pkl` dan `models/ebt_dbp.pkl`, masing-masing
sebagai dict `{"model", "fs", "epoch_len", "feature_names", "target"}`. Saat
inferensi, GUI memvalidasi `fs`/`epoch_len` model terhadap `FS_TARGET`/`EPOCH_LEN`
aktif — model dengan metadata tidak cocok ditolak (tidak dipakai untuk prediksi).

Laporan evaluasi (MAE, RMSE, MAPE, R²) tercetak ke console dan tersimpan ke
`reports/eval_report.csv`. Acuan literatur: MAPE < 2%, R² ≈ 0.99 (tolok ukur,
bukan hard requirement — lihat catatan metodologis di bawah).

### Preprocessing: notch 50Hz + bandpass 2-40Hz

`src/preprocessing/filters.py` menerapkan **notch filter 50Hz** (`iirnotch`,
Q=30, membuang interferensi jala-jala listrik) **sebelum** bandpass Chebyshev
Type II 2-40Hz. Titik masuk tunggalnya adalah `apply_full_preprocessing()`,
dipakai identik oleh `build_features.py` (training) dan `gui/app.py`
(real-time) — tidak ada duplikasi logic filter. Ganti `NOTCH_FREQ_HZ` di
`config.py` ke `60.0` bila akuisisi dilakukan di jaringan listrik 60Hz.

### Verifikasi formula 25 fitur terhadap jurnal rujukan

Seluruh formula fitur di `src/features/time_domain.py` (Table 2, Kandaz &
Uçar 2025) sudah dicek ulang baris-per-baris. Beberapa implementasi
sebelumnya TIDAK sama persis dengan tabel jurnal dan sudah diperbaiki:

| Fitur | Sebelumnya | Sekarang (sesuai jurnal) |
|---|---|---|
| Standard Deviation / Standard Error / CV | std **sampel** (`ddof=1`, `/(n-1)`) | std **populasi** (`ddof=0`, `/n`) — Table 2 #20 eksplisit `S=sqrt((1/n)Σ(x-x̄)²)` |
| Kurtosis / Skewness | kurtosis/skewness populasi biasa (scipy default) | dikali faktor `n/(n-1)` — Table 2 #1,#2 punya faktor `(n-1)` tambahan di penyebut di luar S |
| Average Curve Length | dibagi `n-1` (jumlah suku `diff`) | dibagi `n` — Table 2 #16 eksplisit `(1/n)Σ...` walau sukunya `n-1` |
| Average Teager Energy | dibagi `n-2` (jumlah suku) | dibagi `n` — Table 2 #25 eksplisit `(1/n)Σ...` walau sukunya `n-2` |
| SVD | nilai singular dominan dari trajectory/Hankel matrix (rekayasa sendiri) | norma-2 Euclidean `‖x‖` — persis perilaku `svd(x)` MATLAB pada vektor |
| Trimmed Mean 25%/50% | `scipy.trim_mean(x, 0.25)` / `(x, 0.5)` (yang terakhir selalu NaN) | `scipy.trim_mean(x, 0.125)` / `(x, 0.25)` — konvensi `trimmean(x,percent)` MATLAB: `percent` = total dibuang, dibagi 2 sisi |

Hjorth Mobility/Complexity (#8, #9) di tabel jurnal tercetak tanpa tanda akar
— kemungkinan besar artefak ekstraksi PDF dari tabel berbentuk gambar (bila
dibaca literal tanpa akar, hasilnya tidak konsisten secara dimensi dengan
definisi baku Hjorth 1970). Kode ini **tetap memakai definisi standar**
(dengan akar), bukan versi tercetak — didokumentasikan di komentar
`time_domain.py`. Detail lengkap tiap fitur ada di komentar modul tersebut.

**Konsekuensi:** karena formula fitur berubah, model lama (`models/*.pkl`)
tidak valid lagi dan sudah dilatih ulang dari fitur yang baru diekstraksi.

### Kenapa `BaggingRegressor(estimator=DecisionTreeRegressor())`, bukan `RandomForestRegressor`?

"Ensemble Bagged Tree" secara harfiah adalah bagging atas decision tree biasa
(`BaggingRegressor` + `DecisionTreeRegressor`) — setiap tree dilatih pada
bootstrap sample dengan akses ke SEMUA fitur di tiap split. `RandomForestRegressor`
menambahkan random feature subsampling per split, sebuah varian ensemble yang
berbeda secara definisi, sehingga tidak dipakai di sini.

### Catatan metodologis penting

10-fold CV saat ini membagi fold di level **epoch**, bukan per pasien —
epoch dari pasien yang sama berpeluang muncul di fold latih dan fold uji
sekaligus. Angka MAPE/R² yang dilaporkan kemungkinan optimistis; performa
pada pasien yang benar-benar baru bisa lebih rendah. Pertimbangkan
subject-wise split (`GroupKFold` berbasis indeks pasien) untuk evaluasi yang
lebih jujur sebelum angka ini dipakai di naskah skripsi.

## Menjalankan aplikasi GUI

```powershell
python -m src.gui.app
```

- Akuisisi real-time **hanya lewat sensor Shimmer3R** (tidak ada mode simulasi/
  dummy). Pilih port serial device di dropdown "Port Shimmer3R", klik "Refresh
  Port" bila device belum muncul di daftar, lalu klik "Mulai Pengukuran".
- Logic koneksi Bluetooth/serial di-reuse dari `main_app_fix.py` (kelas
  `ShimmerReader` lama) — lihat `src/acquisition/shimmer_interface.py`. Native
  sample rate device dibaca lewat `get_sampling_rate()` dari pyshimmer, lalu
  setiap sampel di-resample ke `FS_TARGET` sebelum masuk pipeline.
- Setiap epoch 10 detik (1250 sampel pasca-resample) selesai terkumpul:
  notch 50Hz → bandpass 2-40Hz → 25 fitur → prediksi EBT → tampil di panel & tabel histori.
- Histori pengukuran bisa diekspor ke CSV lewat tombol "Ekspor CSV".

## Menjalankan test

```powershell
python -m pytest tests/ -v
```

Mencakup: unit test filter notch+bandpass, 25 fitur (termasuk verifikasi
formula persis terhadap Table 2 jurnal), resampler (native_fs 256/512/1024/511.9 Hz),
epoching, interface Shimmer (bentuk API + konversi ADC→mV, tanpa hardware fisik),
dan test end-to-end (sinyal sintetis → resample → filter → epoch → fitur →
prediksi), termasuk skenario `native_fs != FS_TARGET`. Sinyal sintetis di test
dibangkitkan langsung di dalam masing-masing test (bukan lewat utilitas
produksi) — tidak ada mode dummy di kode aplikasi.

## Keterbatasan

- Mode Shimmer3R belum diuji dengan hardware fisik di lingkungan pengembangan
  ini (tidak tersedia) — logic koneksi di-reuse dari kode lama yang sebelumnya
  terbukti jalan (`main_app_fix.py`), namun jalur resampling native_fs→125Hz di
  `shimmer_interface.py` perlu diverifikasi ulang dengan device fisik.
- Skema evaluasi 10-fold CV saat ini berpotensi data leakage antar-pasien
  (lihat "Catatan metodologis" di atas).
- Hjorth Mobility/Complexity memakai definisi standar (dengan akar) karena
  tabel jurnal tercetak tanpa akar (kemungkinan artefak ekstraksi PDF) — lihat
  "Verifikasi formula 25 fitur" di atas.
