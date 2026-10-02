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

from src.config import FEATURE_REPORT_PATH, EXPERIMENT_LOG_PATH

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


def log_experiment(target_name: str, metrics: dict, run_info: dict, log_path: str = EXPERIMENT_LOG_PATH):
    """Tambahkan (APPEND, tidak pernah menimpa) satu baris riwayat eksperimen.

    Beda dengan print_and_save_report() -- yang menyimpan HANYA hasil model
    TERAKHIR (satu baris per target, ditimpa tiap training ulang) -- fungsi
    ini mengakumulasi SETIAP kali train_and_save() dipanggil, supaya
    beberapa percobaan PARAM_GRID bisa dibandingkan berdampingan di satu
    file histori (urutkan kolom MAPE di Excel/pandas untuk cari yang terbaik).

    run_info wajib berisi key: timestamp, cv_mae, best_params, n_samples,
    n_groups, n_splits, param_grid, features_csv, model_tag, sbp_model_path,
    dbp_model_path (lihat pemanggilnya di src/model/train.py::train_and_save).
    model_tag kosong ("") berarti run itu MENIMPA model produksi (tanpa --tag).

    Gagal menulis (mis. file sedang dibuka di Excel di Windows -> file
    locked) HANYA dicatat sebagai warning, TIDAK melempar exception -- model
    yang sudah berhasil dilatih & disimpan tidak boleh dianggap gagal cuma
    karena baris log gagal ditulis.
    """
    row = {
        "timestamp": run_info["timestamp"],
        "target": target_name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "R2": metrics["R2"],
        "cv_mae": run_info["cv_mae"],
        "best_params": run_info["best_params"],
        "n_samples": run_info["n_samples"],
        "n_groups": run_info["n_groups"],
        "n_splits": run_info["n_splits"],
        "param_grid": run_info["param_grid"],
        "features_csv": run_info["features_csv"],
        "model_tag": run_info["model_tag"],
        "sbp_model_path": run_info["sbp_model_path"],
        "dbp_model_path": run_info["dbp_model_path"],
    }
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        write_header = not os.path.exists(log_path)
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=write_header, index=False)
        logger.info("Eksperimen %s ditambahkan ke %s", target_name, log_path)
    except Exception as e:
        logger.warning(
            "Gagal menulis experiment_log ke %s (%s) -- training TETAP dianggap "
            "berhasil, model sudah tersimpan. Tutup file itu bila sedang terbuka "
            "di aplikasi lain lalu latih ulang bila baris ini perlu tercatat.",
            log_path, e,
        )
