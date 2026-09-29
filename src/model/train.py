"""
Training Ensemble Bagged Tree Regressor (EBT) untuk SBP & DBP.

# Bagian 4.x proposal — Model EBT (Kandaz & Ucar, 2025)

EBT diimplementasikan sebagai sklearn.ensemble.BaggingRegressor dengan
base_estimator=DecisionTreeRegressor — ini SECARA HARFIAH definisi
"Ensemble Bagged Tree" (bagging atas decision tree), bukan RandomForest
(yang menambahkan random feature subsampling per split, sebuah varian
berbeda). Dua model terpisah dilatih (SBP, DBP) karena keduanya punya
karakteristik non-linear yang berbeda terhadap fitur time-domain yang sama,
dan target performa (MAPE/R2) di proposal dilaporkan terpisah per target.

Model disimpan berikut metadata FS_TARGET & EPOCH_LEN supaya saat inferensi
real-time sistem bisa memvalidasi kecocokan sample rate sebelum prediksi.

Validasi memakai GroupKFold (subject-wise), BUKAN KFold biasa. Tiap rekaman
(source_file + record_index, ~22 epoch/rekaman rata-rata) dijadikan satu
grup dan dijamin tidak pernah terpecah antara fold latih & validasi. Tanpa
ini, epoch-epoch dari rekaman/pasien yang sama (yang sangat mirip satu sama
lain) bisa nyasar ke fold train DAN validasi sekaligus -> model "mengintip"
data dari subjek yang sama saat divalidasi -> metrik (MAE/RMSE/MAPE/R2)
jadi bias optimis, tidak mencerminkan kemampuan generalisasi ke pasien baru
yang sesungguhnya jadi skenario pemakaian sistem ini.
"""

import argparse
import itertools
import logging
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    FS_TARGET,
    EPOCH_LEN,
    FEATURE_NAMES,
    PROCESSED_DATA_DIR,
    SBP_MODEL_PATH,
    DBP_MODEL_PATH,
)
from src.model.evaluate import compute_metrics, log_experiment, print_and_save_report

# stdout (print) dan stderr (logger) dibuat line-buffered supaya keduanya
# tampil ke terminal dalam urutan kronologis yang benar, bukan tertahan di
# buffer sampai proses selesai atau bercampur acak saat di-pipe/redirect.
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Grid parameter untuk GridSearchCV. GridSearchCV mencoba SETIAP kombinasi
# nilai di bawah ini x n_splits fold, jadi ukuran grid menentukan waktu
# training secara langsung (lihat _print_param_grid saat dijalankan untuk
# jumlah fit & progresnya; parameter yang benar-benar terpilih tersimpan di
# models/ebt_{sbp,dbp}.pkl dan bisa dibaca lewat model["model"].get_params(),
# BUKAN di komentar ini -- jangan catat hasil eksperimen di sini, catat di
# reports/ atau pesan commit, supaya komentar ini tidak basi tiap kali grid
# diubah).
#
# Arti tiap parameter (BaggingRegressor + DecisionTreeRegressor):
#   - n_estimators: jumlah pohon yang dirata-rata. Bagging tidak overfit
#     dengan menambah pohon -- ini murni tukar waktu training dengan
#     stabilitas prediksi, manfaatnya mengecil cepat setelah puluhan pohon.
#   - estimator__min_samples_leaf: ukuran minimum daun tiap pohon. Kecil
#     (mis. 1) -> pohon bisa menghafal sampel individual (overfit); besar
#     -> prediksi lebih halus tapi bisa terlalu kasar (underfit). Ini
#     parameter yang paling menentukan trade-off overfit/underfit.
#   - max_samples: fraksi data (dengan pengembalian) yang dilihat tiap
#     pohon. Lebih kecil -> pohon lebih beragam & lebih cepat dilatih.
#   - estimator__max_depth: kedalaman maksimum pohon (tidak dipakai di grid
#     saat ini). Efeknya tumpang tindih dengan min_samples_leaf; tambahkan
#     lagi hanya kalau min_samples_leaf saja belum cukup meregulasi pohon.
PARAM_GRID = {
    "n_estimators": [60],
    "estimator__min_samples_leaf": [8, 30, 100],
    "max_samples": [1.0],
}
RANDOM_STATE = 42


def _print_cv_scheme(n_samples: int, groups: np.ndarray, n_splits: int, random_state: int):
    """Cetak rincian pembagian data latih/validasi per fold secara eksplisit,
    dihitung dari GroupKFold.split() yang sungguhan dipakai (bukan
    estimasi/asumsi), supaya bisa dipertanggungjawabkan angka pastinya per
    fold -- termasuk jumlah subjek (pasien/rekaman) di tiap fold, bukan cuma
    jumlah epoch, karena inilah yang dijamin GroupKFold: subjek yang sama
    tidak pernah terpecah antara train & validasi."""
    n_groups = len(np.unique(groups))
    group_kfold = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print("\n" + "=" * 60)
    print("SKEMA VALIDASI: GROUP K-FOLD CROSS-VALIDATION (SUBJECT-WISE)")
    print("=" * 60)
    print(f"Total data (epoch)      : {n_samples}")
    print(f"Jumlah subjek unik      : {n_groups}  (grup = source_file + record_index)")
    print(f"Rata-rata epoch/subjek  : {n_samples / n_groups:.1f}")
    print(f"Jumlah fold (K)         : {n_splits}")
    print(f"shuffle                 : True")
    print(f"random_state            : {random_state}  (agar pembagian fold reproducible)")
    print("-" * 60)
    print("PENTING: satu subjek (pasien/rekaman) HANYA pernah muncul di SALAH")
    print("SATU sisi (train ATAU validasi) dalam satu fold, tidak pernah dua-duanya.")
    print("Ini mencegah model 'mengintip' data dari subjek yang sama saat divalidasi.")
    print("-" * 60)
    print(f"{'Fold':<6}{'Data latih (train)':<24}{'Data validasi (val)':<24}")
    for i, (train_idx, val_idx) in enumerate(group_kfold.split(np.zeros(n_samples), groups=groups), start=1):
        train_pct = len(train_idx) / n_samples * 100
        val_pct = len(val_idx) / n_samples * 100
        print(f"{i:<6}{f'{len(train_idx)} ({train_pct:.1f}%)':<24}{f'{len(val_idx)} ({val_pct:.1f}%)':<24}")
    print("-" * 60)
    print("Catatan: TIDAK ada split tetap 80:20. Rasio per putaran mendekati")
    print(f"~{(n_splits-1)/n_splits*100:.0f}:{100/n_splits:.0f} (train:val) -- tidak persis sama tiap fold karena")
    print("ukuran grup (epoch/subjek) tidak seragam. Diulang %d x sampai setiap" % n_splits)
    print("subjek pernah menjadi data validasi tepat 1 kali (cross_val_predict).")
    print("=" * 60 + "\n")


def _fmt_param_value(v):
    return "None (tanpa batas)" if v is None else str(v)


def _print_param_grid(target_name: str, n_splits: int):
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    print("\n" + "=" * 60)
    print(f"GRIDSEARCHCV - PARAMETER YANG DIUJI UNTUK TARGET: {target_name}")
    print("=" * 60)
    print(f"Model dasar          : BaggingRegressor(estimator=DecisionTreeRegressor())")
    for key in keys:
        print(f"{key:<21}: {PARAM_GRID[key]}")
    print(f"scoring              : neg_mean_absolute_error (MAE terkecil menang)")
    print(f"random_state         : {RANDOM_STATE}")
    print("-" * 60)
    print(f"Kombinasi yang akan dicoba ({len(combos)} kombinasi x {n_splits} fold = {len(combos)*n_splits} fit):")
    for i, combo in enumerate(combos, start=1):
        combo_str = ", ".join(f"{k}={_fmt_param_value(v)}" for k, v in zip(keys, combo))
        print(f"  {i}. {combo_str}")
    print("=" * 60 + "\n")


def train_single_target(X: np.ndarray, y: np.ndarray, groups: np.ndarray, target_name: str = "", n_splits: int = 10, random_state: int = 42):
    _print_param_grid(target_name, n_splits)

    base = DecisionTreeRegressor(random_state=random_state)
    bagging = BaggingRegressor(estimator=base, random_state=random_state)

    # GroupKFold: pembagian fold memakai groups (subjek/rekaman), bukan index
    # baris polos -- lihat docstring modul untuk alasannya (cegah data leakage
    # antar-epoch dari subjek yang sama).
    group_kfold = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_combos = len(list(itertools.product(*PARAM_GRID.values())))
    logger.info(
        "GridSearchCV: %d kombinasi param x %d fold = %d fit total (n_jobs=-1, paralel)",
        n_combos, n_splits, n_combos * n_splits,
    )
    # verbose=3 -> setiap fit selesai dicetak ke console (durasi, kombinasi param,
    # skor) supaya progres kelihatan live, bukan diam sampai semuanya kelar.
    search = GridSearchCV(
        bagging, PARAM_GRID, cv=group_kfold, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=3
    )
    search.fit(X, y, groups=groups)

    best_params_str = ", ".join(f"{k}={_fmt_param_value(v)}" for k, v in search.best_params_.items())
    print(f"\n>>> Parameter TERPILIH untuk {target_name}: {best_params_str}  "
          f"(CV MAE={-search.best_score_:.4f})\n")
    logger.info("Best params: %s (CV MAE=%.3f)", search.best_params_, -search.best_score_)

    # Prediksi out-of-fold GroupKFold dengan model terbaik untuk evaluasi tak-bias
    # (subject-wise -- tidak ada subjek yang bocor antara train & validasi)
    logger.info("Menghitung prediksi out-of-fold (cross_val_predict, %d fold, subject-wise)...", n_splits)
    y_pred_cv = cross_val_predict(search.best_estimator_, X, y, groups=groups, cv=group_kfold, n_jobs=-1, verbose=3)

    return search.best_estimator_, y_pred_cv, search.best_params_, -search.best_score_


def train_and_save(features_csv: str, n_splits: int = 10):
    df = pd.read_csv(features_csv).dropna()
    logger.info("Dataset fitur: %d baris, %d kolom", *df.shape)

    if "source_file" not in df.columns or "record_index" not in df.columns:
        raise ValueError(
            "Kolom 'source_file'/'record_index' tidak ditemukan di features_csv -- "
            "dibutuhkan sebagai ID subjek untuk GroupKFold (subject-wise CV). "
            "Regenerasi features.csv lewat: python -m src.data_pipeline.build_features"
        )

    X = df[FEATURE_NAMES].values
    y_sbp = df["SBP"].values
    y_dbp = df["DBP"].values
    # ID subjek/rekaman unik: satu rekaman ECG (source_file+record_index)
    # menyumbang banyak epoch yang saling mirip -> semuanya harus tetap di
    # sisi fold yang sama (lihat docstring modul / GroupKFold di atas).
    groups = (df["source_file"].astype(str) + "::" + df["record_index"].astype(str)).values

    _print_cv_scheme(n_samples=len(df), groups=groups, n_splits=n_splits, random_state=RANDOM_STATE)

    # Timestamp BERSAMA untuk baris SBP & DBP di experiment_log.csv, supaya
    # keduanya mudah dikenali berasal dari run yang sama saat log dibaca ulang.
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_info_common = {
        "timestamp": run_timestamp,
        "n_samples": len(df),
        "n_groups": len(np.unique(groups)),
        "n_splits": n_splits,
        "param_grid": str(PARAM_GRID),
        "features_csv": features_csv,
    }

    logger.info("Melatih model SBP (GroupKFold subject-wise + GridSearchCV)...")
    model_sbp, y_sbp_pred_cv, best_params_sbp, cv_mae_sbp = train_single_target(
        X, y_sbp, groups, target_name="SBP", n_splits=n_splits, random_state=RANDOM_STATE
    )
    # Dicatat SEGERA setelah SBP selesai (bukan menunggu DBP) -- kalau training
    # DBP gagal/di-Ctrl+C setelahnya, hasil eksperimen SBP tidak ikut hilang.
    log_experiment(
        "SBP", compute_metrics(y_sbp, y_sbp_pred_cv),
        {**run_info_common, "best_params": str(best_params_sbp), "cv_mae": cv_mae_sbp},
    )

    logger.info("Melatih model DBP (GroupKFold subject-wise + GridSearchCV)...")
    model_dbp, y_dbp_pred_cv, best_params_dbp, cv_mae_dbp = train_single_target(
        X, y_dbp, groups, target_name="DBP", n_splits=n_splits, random_state=RANDOM_STATE
    )
    log_experiment(
        "DBP", compute_metrics(y_dbp, y_dbp_pred_cv),
        {**run_info_common, "best_params": str(best_params_dbp), "cv_mae": cv_mae_dbp},
    )

    metadata_sbp = {
        "model": model_sbp,
        "fs": FS_TARGET,
        "epoch_len": EPOCH_LEN,
        "feature_names": FEATURE_NAMES,
        "target": "SBP",
    }
    metadata_dbp = {
        "model": model_dbp,
        "fs": FS_TARGET,
        "epoch_len": EPOCH_LEN,
        "feature_names": FEATURE_NAMES,
        "target": "DBP",
    }

    joblib.dump(metadata_sbp, SBP_MODEL_PATH)
    joblib.dump(metadata_dbp, DBP_MODEL_PATH)
    logger.info("Model tersimpan: %s, %s", SBP_MODEL_PATH, DBP_MODEL_PATH)

    return {
        "y_sbp": y_sbp, "y_sbp_pred": y_sbp_pred_cv,
        "y_dbp": y_dbp, "y_dbp_pred": y_dbp_pred_cv,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih model EBT untuk SBP & DBP.")
    parser.add_argument("--features-csv", default=f"{PROCESSED_DATA_DIR}/features.csv")
    parser.add_argument("--n-splits", type=int, default=10)
    args = parser.parse_args()

    results = train_and_save(args.features_csv, n_splits=args.n_splits)

    print_and_save_report(results["y_sbp"], results["y_sbp_pred"], "SBP")
    print_and_save_report(results["y_dbp"], results["y_dbp_pred"], "DBP")
