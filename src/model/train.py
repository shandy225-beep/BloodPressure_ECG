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
"""

import argparse
import itertools
import logging
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    FS_TARGET,
    EPOCH_LEN,
    FEATURE_NAMES,
    PROCESSED_DATA_DIR,
    SBP_MODEL_PATH,
    DBP_MODEL_PATH,
)

# stdout (print) dan stderr (logger) dibuat line-buffered supaya keduanya
# tampil ke terminal dalam urutan kronologis yang benar, bukan tertahan di
# buffer sampai proses selesai atau bercampur acak saat di-pipe/redirect.
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Grid ringan sesuai catatan "jangan over-engineer, fokus baseline solid dulu"
PARAM_GRID = {
    "n_estimators": [30, 60],
    "estimator__max_depth": [6, 10, None],
}
RANDOM_STATE = 42


def _print_cv_scheme(n_samples: int, n_splits: int, random_state: int):
    """Cetak rincian pembagian data latih/validasi per fold secara eksplisit,
    dihitung dari KFold.split() yang sungguhan dipakai (bukan estimasi/asumsi),
    supaya bisa dipertanggungjawabkan angka pastinya per fold."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print("\n" + "=" * 60)
    print("SKEMA VALIDASI: K-FOLD CROSS-VALIDATION")
    print("=" * 60)
    print(f"Total data (epoch)      : {n_samples}")
    print(f"Jumlah fold (K)         : {n_splits}")
    print(f"shuffle                 : True")
    print(f"random_state            : {random_state}  (agar pembagian fold reproducible)")
    print("-" * 60)
    print(f"{'Fold':<6}{'Data latih (train)':<24}{'Data validasi (val)':<24}")
    for i, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(n_samples)), start=1):
        train_pct = len(train_idx) / n_samples * 100
        val_pct = len(val_idx) / n_samples * 100
        print(f"{i:<6}{f'{len(train_idx)} ({train_pct:.1f}%)':<24}{f'{len(val_idx)} ({val_pct:.1f}%)':<24}")
    print("-" * 60)
    print("Catatan: TIDAK ada split tetap 80:20. Rasio per putaran adalah")
    print(f"~{(n_splits-1)/n_splits*100:.0f}:{100/n_splits:.0f} (train:val), diulang {n_splits}x sampai setiap")
    print("epoch pernah menjadi data validasi tepat 1 kali (cross_val_predict).")
    print("=" * 60 + "\n")


def _print_param_grid(target_name: str, n_splits: int):
    combos = list(itertools.product(PARAM_GRID["n_estimators"], PARAM_GRID["estimator__max_depth"]))
    print("\n" + "=" * 60)
    print(f"GRIDSEARCHCV - PARAMETER YANG DIUJI UNTUK TARGET: {target_name}")
    print("=" * 60)
    print(f"Model dasar          : BaggingRegressor(estimator=DecisionTreeRegressor())")
    print(f"n_estimators (grid)  : {PARAM_GRID['n_estimators']}")
    print(f"max_depth tree (grid): {PARAM_GRID['estimator__max_depth']}")
    print(f"scoring              : neg_mean_absolute_error (MAE terkecil menang)")
    print(f"random_state         : {RANDOM_STATE}")
    print("-" * 60)
    print(f"Kombinasi yang akan dicoba ({len(combos)} kombinasi x {n_splits} fold = {len(combos)*n_splits} fit):")
    for i, (n_est, depth) in enumerate(combos, start=1):
        depth_str = "None (tanpa batas)" if depth is None else str(depth)
        print(f"  {i}. n_estimators={n_est:<4} max_depth={depth_str}")
    print("=" * 60 + "\n")


def train_single_target(X: np.ndarray, y: np.ndarray, target_name: str = "", n_splits: int = 10, random_state: int = 42):
    _print_param_grid(target_name, n_splits)

    base = DecisionTreeRegressor(random_state=random_state)
    bagging = BaggingRegressor(estimator=base, random_state=random_state)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_combos = len(PARAM_GRID["n_estimators"]) * len(PARAM_GRID["estimator__max_depth"])
    logger.info(
        "GridSearchCV: %d kombinasi param x %d fold = %d fit total (n_jobs=-1, paralel)",
        n_combos, n_splits, n_combos * n_splits,
    )
    # verbose=3 -> setiap fit selesai dicetak ke console (durasi, kombinasi param,
    # skor) supaya progres kelihatan live, bukan diam sampai semuanya kelar.
    search = GridSearchCV(
        bagging, PARAM_GRID, cv=kfold, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=3
    )
    search.fit(X, y)

    depth_chosen = search.best_params_["estimator__max_depth"]
    depth_str = "None (tanpa batas)" if depth_chosen is None else str(depth_chosen)
    print(f"\n>>> Parameter TERPILIH untuk {target_name}: "
          f"n_estimators={search.best_params_['n_estimators']}, "
          f"max_depth={depth_str}  (CV MAE={-search.best_score_:.4f})\n")
    logger.info("Best params: %s (CV MAE=%.3f)", search.best_params_, -search.best_score_)

    # Prediksi out-of-fold 10-fold CV dengan model terbaik untuk evaluasi tak-bias
    logger.info("Menghitung prediksi out-of-fold (cross_val_predict, %d fold)...", n_splits)
    y_pred_cv = cross_val_predict(search.best_estimator_, X, y, cv=kfold, n_jobs=-1, verbose=3)

    return search.best_estimator_, y_pred_cv


def train_and_save(features_csv: str, n_splits: int = 10):
    df = pd.read_csv(features_csv).dropna()
    logger.info("Dataset fitur: %d baris, %d kolom", *df.shape)

    X = df[FEATURE_NAMES].values
    y_sbp = df["SBP"].values
    y_dbp = df["DBP"].values

    _print_cv_scheme(n_samples=len(df), n_splits=n_splits, random_state=RANDOM_STATE)

    logger.info("Melatih model SBP (10-fold CV + GridSearchCV)...")
    model_sbp, y_sbp_pred_cv = train_single_target(X, y_sbp, target_name="SBP", n_splits=n_splits, random_state=RANDOM_STATE)

    logger.info("Melatih model DBP (10-fold CV + GridSearchCV)...")
    model_dbp, y_dbp_pred_cv = train_single_target(X, y_dbp, target_name="DBP", n_splits=n_splits, random_state=RANDOM_STATE)

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

    from src.model.evaluate import print_and_save_report

    print_and_save_report(results["y_sbp"], results["y_sbp_pred"], "SBP")
    print_and_save_report(results["y_dbp"], results["y_dbp_pred"], "DBP")
