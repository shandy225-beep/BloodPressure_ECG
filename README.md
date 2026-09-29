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
│   │   ├── train.py                 # EBT (BaggingRegressor) + GroupKFold 10-fold (subject-wise) + GridSearchCV
│   │   └── evaluate.py              # MAE, RMSE, MAPE, R²
│   ├── acquisition/
│   │   ├── resampler.py             # StreamingResampler (real-time) + resample_to_target() (offline)
│   │   └── shimmer_interface.py     # koneksi Shimmer3R (reuse main_app_fix.py)
│   └── gui/app.py                   # aplikasi real-time PyQt5 + pyqtgraph
├── models/                          # ebt_sbp.pkl, ebt_dbp.pkl (+ metadata FS)
├── reports/eval_report.csv
├── evenv/                           # virtual environment (Python 3.14.5)
└── tests/
```

Catatan: dataset mentah (`part_N.mat`) dirujuk langsung dari `Dataset/` alih-alih
disalin ke `data/raw/` — file-nya ratusan MB per part, jadi `RAW_DATA_DIR` di
`config.py` menunjuk langsung ke `Dataset/` (seluruh `part_*.mat` ditemukan
otomatis lewat `DATASET_GLOB_PATTERN`).

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

# 2. Latih model EBT (BaggingRegressor + GridSearchCV atas grid di
#    src/model/train.py, PARAM_GRID). Validasi memakai GroupKFold subject-wise
#    (default 10 fold, ganti dengan --n-splits) -- lihat "Catatan metodologis"
#    di bawah untuk kenapa ini penting. Dengan seluruh 12 part, jumlah baris
#    fitur jauh lebih besar dari part_1 saja (~25 ribu epoch) — proses
#    training proporsional lebih lama juga.
python -m src.model.train --features-csv data/processed/features.csv
```

Model tersimpan ke `models/ebt_sbp.pkl` dan `models/ebt_dbp.pkl`, masing-masing
sebagai dict `{"model", "fs", "epoch_len", "feature_names", "target"}`. Saat
inferensi, GUI memvalidasi `fs`/`epoch_len` model terhadap `FS_TARGET`/`EPOCH_LEN`
aktif — model dengan metadata tidak cocok ditolak (tidak dipakai untuk prediksi).
Parameter yang benar-benar terpilih oleh `GridSearchCV` (bisa berubah tiap kali
`PARAM_GRID` diubah atau ditraining ulang) tersimpan di file model itu sendiri,
dibaca lewat `joblib.load(path)["model"].get_params()` — bukan di README ini.

Laporan evaluasi (MAE, RMSE, MAPE, R²) tercetak ke console dan tersimpan ke
`reports/eval_report.csv`. Acuan literatur: MAPE < 2%, R² ≈ 0.99 (tolok ukur,
bukan hard requirement, dan kemungkinan tidak apple-to-apple dengan evaluasi di
proyek ini — lihat "Catatan metodologis" di bawah).

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

### Catatan metodologis penting: evaluasi subject-wise (`GroupKFold`)

Validasi memakai `GroupKFold` (`src/model/train.py`), BUKAN `KFold` biasa.
Grup = satu rekaman (`source_file` + `record_index`, rata-rata ~24 epoch per
rekaman). `GroupKFold` menjamin satu rekaman/pasien **hanya pernah muncul di
salah satu sisi** (latih ATAU validasi) dalam satu fold, tidak pernah dua-duanya.

**Kenapa ini penting, bukan sekadar pilihan teknis:** epoch-epoch dari satu
rekaman sangat mirip satu sama lain (SBP/DBP hampir sama di semua epoch
rekaman itu). Pada data fitur proyek ini, sekitar **90% variasi SBP/DBP
adalah variasi ANTAR-pasien**, bukan dalam satu pasien. Kalau fold dibagi
per-epoch (`KFold` biasa, cara lama), epoch dari pasien yang sama bisa
muncul di fold latih DAN fold uji sekaligus — model bisa "mengenali pasien
itu" dari epoch tetangganya alih-alih benar-benar menaksir dari bentuk
sinyal ECG, sehingga metrik yang dilaporkan bias optimistis dan tidak
mencerminkan performa pada pasien baru. Di literatur, perbedaan ini dikenal
sebagai evaluasi *record-wise*/*intra-patient* (cara lama) vs
*subject-wise*/*inter-patient* (`GroupKFold`, dipakai sekarang) — lihat
dokumentasi scikit-learn bagian
["Cross-validation iterators for grouped data"](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data),
serta Saeb et al. (2017, *GigaScience* 6(5)) dan Tougui et al. (2021,
*Healthcare Informatics Research*) yang membahas dampak pilihan ini pada
model diagnostik klinis.

**Dampaknya terukur langsung di proyek ini.** Sebelum `GroupKFold` dipasang,
`reports/eval_report.csv` melaporkan (evaluasi per-epoch, BOCOR antar-pasien):

| Target | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| SBP | 5.81 | 8.97 | 4.69% | 0.836 |
| DBP | 3.29 | 5.71 | 4.68% | 0.774 |

Setelah dipindah ke `GroupKFold` subject-wise (angka jujur, pasien uji tidak
pernah dilihat model saat latihan):

| Target | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| SBP | 10.65 | 15.05 | 8.65% | 0.540 |
| DBP | 5.80 | 8.75 | 8.35% | 0.468 |

Sebagai pembanding, menebak rata-rata populasi untuk semua pasien (tanpa
memakai ECG sama sekali) menghasilkan MAE≈18.0 (SBP) dan MAE≈9.25 (DBP) —
jadi model subject-wise di atas tetap memangkas galat sekitar 40% dibanding
menebak buta, tetapi **jauh dari target literatur (MAPE<2%, MAE~10 mmHg
di atas target)**.

**Implikasi untuk penulisan skripsi:**
- Angka subject-wise di atas adalah yang mencerminkan kemampuan sistem pada
  **pasien baru** — inilah angka yang seharusnya jadi hasil utama, bukan
  angka per-epoch yang bocor.
- Selisih besar dengan target jurnal (`MAPE<2%, R²≈0.99`) kemungkinan
  sebagian berasal dari perbedaan protokol validasi (jurnal rujukan memakai
  jumlah subjek yang jauh lebih kecil dan mungkin evaluasi record-wise —
  BELUM diverifikasi langsung dari naskah jurnal, cek bagian metodenya)
  dan/atau keterbatasan informasi 25 fitur time-domain non-fiducial dari
  ECG saja (tanpa sinyal kedua seperti PPG, tidak ada informasi pulse
  transit time).
- `GridSearchCV` memilih parameter DAN metrik akhir dilaporkan dari fold
  yang sama, sehingga angka di atas sedikit optimistis. Untuk laporan
  tingkat skripsi, sisihkan sekitar 20% pasien sebagai data uji terpisah
  yang tidak disentuh sama sekali saat pemilihan parameter.

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
  setiap sampel di-resample ke `FS_TARGET` lewat `StreamingResampler`
  (stateful, tidak menimbulkan artefak di batas potongan stream — beda
  dengan memanggil `resample_to_target()` berulang pada potongan pendek).
- Sampel yang sudah di-resample masuk ke `FilteredEpochBuffer`
  (`src/data_pipeline/epoching.py`): memfilter jendela epoch + konteks
  kiri/kanan (`FILTER_CONTEXT_SEC`, default 3 detik) lalu memotong bagian
  tengahnya, supaya epoch yang di-ekstrak fiturnya SETARA dengan cara
  training (filter seluruh rekaman lalu potong epoch) — bukan filter tiap
  epoch 10 detik secara terpisah, yang menimbulkan efek tepi zero-phase dan
  membuat fitur bergeser jauh dari distribusi training. Konsekuensinya, tiap
  estimasi SBP/DBP baru muncul sekitar `FILTER_CONTEXT_SEC` detik setelah
  epoch 10 detiknya selesai terkumpul (butuh sampel "masa depan" untuk sisi
  kanan filter).
- Plot ECG live memakai filter zero-phase yang SAMA (`apply_full_preprocessing`)
  pada jendela geser, sehingga tampilan tertunda `GUI_DISPLAY_LAG_SEC` (default
  1 detik) dari sampel terbaru — lihat komentar di `config.py` untuk alasan
  filter kausal tanpa delay ditolak (mendistorsi morfologi QRS).
- Histori pengukuran bisa diekspor ke CSV lewat tombol "Ekspor CSV".

## Menjalankan test

```powershell
python -m pytest tests/ -v
```

Mencakup: unit test filter notch+bandpass, 25 fitur (termasuk verifikasi
formula persis terhadap Table 2 jurnal), epoching dasar (`segment_epochs`,
target SBP/DBP dari ABP), `FilteredEpochBuffer` (epoch streaming setara
filter-seluruh-rekaman, diverifikasi terhadap `apply_full_preprocessing`
pada sinyal utuh), `resample_to_target` offline (native_fs
256/512/1024/511.9 Hz), `StreamingResampler` (identik `resample_poly` di
bagian interior, hasil independen dari ukuran potongan yang disuapkan, tanpa
drift timing untuk native_fs pecahan), interface Shimmer (bentuk API,
konversi ADC→mV, alur paket→resampler→queue lewat paket palsu, tanpa
hardware fisik), plot GUI live (offscreen Qt: kurva benar-benar terfilter,
sumbu waktu berbasis sampel, judul mencerminkan konfigurasi aktif), dan test
end-to-end (sinyal sintetis → resample → epoch terfilter → fitur →
prediksi), termasuk skenario `native_fs != FS_TARGET`. Sinyal sintetis di
test dibangkitkan langsung di dalam masing-masing test (bukan lewat utilitas
produksi) — tidak ada mode dummy di kode aplikasi.

## Keterbatasan

- Mode Shimmer3R belum diuji dengan hardware fisik di lingkungan pengembangan
  ini (tidak tersedia) — logic koneksi di-reuse dari kode lama yang sebelumnya
  terbukti jalan (`main_app_fix.py`), namun jalur resampling native_fs→125Hz di
  `shimmer_interface.py` perlu diverifikasi ulang dengan device fisik.
- Akurasi estimasi (evaluasi subject-wise, lihat "Catatan metodologis" di
  atas) masih jauh dari target literatur: MAPE 8.65% (SBP) / 8.35% (DBP),
  dibanding acuan MAPE<2%. Ini kemungkinan bukan bug kode, melainkan batas
  informasi 25 fitur time-domain non-fiducial dari ECG single-lead saja
  untuk menaksir tekanan darah pada pasien yang benar-benar baru — belum
  cukup untuk dipakai sebagai alat ukur, meski lebih baik ~40% daripada
  menebak rata-rata populasi (lihat tabel di "Catatan metodologis").
- Fitur time-domain (`src/features/time_domain.py`) memakai amplitudo
  mentah sinyal ECG (mean, max, min, IQR, trimmed mean, dst. tidak
  dinormalisasi per-epoch). Amplitudo ECG antar-rekaman di dataset berbeda
  signifikan, dan gain akuisisi Shimmer3R bisa berbeda dari dataset latih —
  sensitivitas model terhadap perbedaan skala amplitudo ini belum diuji
  langsung dengan data Shimmer.
- Hjorth Mobility/Complexity memakai definisi standar (dengan akar) karena
  tabel jurnal tercetak tanpa akar (kemungkinan artefak ekstraksi PDF) — lihat
  "Verifikasi formula 25 fitur" di atas.
