"""
Rangkaian pipeline penuh: raw ECG/ABP (125Hz) -> filter bandpass -> epoching
-> ekstraksi 25 fitur -> DataFrame siap latih (kolom fitur + SBP + DBP).

Dipakai oleh src/model/train.py dan bisa dijalankan langsung sebagai script
untuk menghasilkan data/processed/features.csv dari dataset asli
(Dataset/part_N.mat).
"""

import argparse
import logging

import pandas as pd

from src.config import FS_TARGET, DEFAULT_DATASET_FILE, PROCESSED_DATA_DIR
from src.preprocessing.filters import apply_bandpass_cheby2
from src.features.time_domain import extract_features
from src.data_pipeline.load_dataset import load_mat_records
from src.data_pipeline.epoching import build_epoch_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_feature_dataframe(mat_path: str = DEFAULT_DATASET_FILE, max_records: int = None) -> pd.DataFrame:
    rows = []

    n_processed = 0
    for ecg_raw, abp_raw in load_mat_records(mat_path):
        if max_records is not None and n_processed >= max_records:
            break

        ecg_filtered = apply_bandpass_cheby2(ecg_raw, fs=FS_TARGET)

        for ecg_epoch, sbp, dbp in build_epoch_dataset(ecg_filtered, abp_raw, fs=FS_TARGET):
            feats = extract_features(ecg_epoch)
            feats["SBP"] = sbp
            feats["DBP"] = dbp
            rows.append(feats)

        n_processed += 1
        if n_processed % 50 == 0:
            logger.info("%d rekaman diproses, %d epoch terkumpul...", n_processed, len(rows))

    logger.info("Selesai: %d rekaman diproses, total %d epoch valid.", n_processed, len(rows))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangun dataset fitur dari sinyal ECG/ABP mentah (dataset asli).")
    parser.add_argument("--mat-path", default=DEFAULT_DATASET_FILE)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = build_feature_dataframe(args.mat_path, args.max_records)

    out_path = args.out or f"{PROCESSED_DATA_DIR}/features.csv"
    df.to_csv(out_path, index=False)
    logger.info("Dataset fitur disimpan ke %s (%d baris)", out_path, len(df))
