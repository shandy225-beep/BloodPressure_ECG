"""
Evaluasi model: MAE, RMSE, MAPE, R^2.

Target acuan literatur (Kandaz & Ucar, 2025): MAPE < 2%, R^2 ~ 0.99 —
bukan hard requirement, hanya tolok ukur pembanding.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import REPORTS_DIR, FEATURE_REPORT_PATH

logger = logging.getLogger(__name__)

MAPE_REFERENCE = 2.0
R2_REFERENCE = 0.99


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    nonzero = y_true != 0
    if np.any(nonzero):
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        logger.warning("Semua y_true bernilai 0, MAPE tidak terdefinisi -> NaN")
        mape = float("nan")

    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def print_and_save_report(y_true: np.ndarray, y_pred: np.ndarray, target_name: str, out_path: str = FEATURE_REPORT_PATH):
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n=== Laporan Evaluasi — {target_name} ===")
    print(f"{'Metrik':<10}{'Nilai':>12}{'Acuan Literatur':>20}")
    print(f"{'MAE':<10}{metrics['MAE']:>12.3f}{'-':>20}")
    print(f"{'RMSE':<10}{metrics['RMSE']:>12.3f}{'-':>20}")
    print(f"{'MAPE (%)':<10}{metrics['MAPE']:>12.3f}{f'< {MAPE_REFERENCE}':>20}")
    print(f"{'R2':<10}{metrics['R2']:>12.4f}{f'~ {R2_REFERENCE}':>20}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    row = pd.DataFrame([{"target": target_name, **metrics}])

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        existing = existing[existing["target"] != target_name]
        row = pd.concat([existing, row], ignore_index=True)

    row.to_csv(out_path, index=False)
    logger.info("Laporan evaluasi %s disimpan ke %s", target_name, out_path)

    return metrics
