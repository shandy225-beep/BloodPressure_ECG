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
import logging

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Grid ringan sesuai catatan "jangan over-engineer, fokus baseline solid dulu"
PARAM_GRID = {
    "n_estimators": [30, 60],
    "estimator__max_depth": [6, 10, None],
}


def train_single_target(X: np.ndarray, y: np.ndarray, n_splits: int = 10, random_state: int = 42):
    base = DecisionTreeRegressor(random_state=random_state)
    bagging = BaggingRegressor(estimator=base, random_state=random_state)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(bagging, PARAM_GRID, cv=kfold, scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(X, y)

    logger.info("Best params: %s (CV MAE=%.3f)", search.best_params_, -search.best_score_)

    # Prediksi out-of-fold 10-fold CV dengan model terbaik untuk evaluasi tak-bias
    y_pred_cv = cross_val_predict(search.best_estimator_, X, y, cv=kfold, n_jobs=-1)

    return search.best_estimator_, y_pred_cv


def train_and_save(features_csv: str, n_splits: int = 10):
    df = pd.read_csv(features_csv).dropna()
    logger.info("Dataset fitur: %d baris, %d kolom", *df.shape)

    X = df[FEATURE_NAMES].values
    y_sbp = df["SBP"].values
    y_dbp = df["DBP"].values

    logger.info("Melatih model SBP (10-fold CV + GridSearchCV)...")
    model_sbp, y_sbp_pred_cv = train_single_target(X, y_sbp, n_splits=n_splits)

    logger.info("Melatih model DBP (10-fold CV + GridSearchCV)...")
    model_dbp, y_dbp_pred_cv = train_single_target(X, y_dbp, n_splits=n_splits)

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
