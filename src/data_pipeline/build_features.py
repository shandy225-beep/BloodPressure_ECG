"""
Rangkaian pipeline penuh: raw ECG/ABP (125Hz) -> notch 50Hz -> bandpass
2-40Hz -> epoching -> ekstraksi 25 fitur -> DataFrame siap latih (kolom
fitur + SBP + DBP).

Dipakai oleh src/model/train.py dan bisa dijalankan langsung sebagai script
untuk menghasilkan data/processed/features.csv dari dataset asli
(Dataset/part_N.mat). Default: memproses SELURUH part_1.mat..part_12.mat
yang ada di Dataset/ (lihat --mat-path untuk memproses satu file saja).
"""

import argparse
import glob
import logging
import re
from typing import List, Union

import pandas as pd

from src.config import FS_TARGET, DEFAULT_DATASET_FILE, DATASET_GLOB_PATTERN, PROCESSED_DATA_DIR
from src.preprocessing.filters import apply_full_preprocessing
from src.features.time_domain import extract_features
from src.data_pipeline.load_dataset import load_mat_records
from src.data_pipeline.epoching import build_epoch_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _part_number(path: str) -> int:
    match = re.search(r"part_(\d+)\.mat$", path)
    return int(match.group(1)) if match else 0


def discover_dataset_files(pattern: str = DATASET_GLOB_PATTERN) -> List[str]:
    """Cari seluruh part_N.mat di Dataset/, diurutkan numerik (part_1, part_2,
    ..., part_12) alih-alih leksikografis (yang akan menaruh part_10 sebelum
    part_2)."""
    files = glob.glob(pattern)
    return sorted(files, key=_part_number)


def build_feature_dataframe(mat_paths: Union[str, List[str]] = DEFAULT_DATASET_FILE, max_records: int = None) -> pd.DataFrame:
    if isinstance(mat_paths, str):
        mat_paths = [mat_paths]

    rows = []
    n_processed = 0

    for mat_path in mat_paths:
        for record_idx, (ecg_raw, abp_raw) in enumerate(load_mat_records(mat_path)):
            if max_records is not None and n_processed >= max_records:
                logger.info("Batas max_records=%d tercapai, berhenti.", max_records)
                return pd.DataFrame(rows)

            ecg_filtered = apply_full_preprocessing(ecg_raw, fs=FS_TARGET)

            for ecg_epoch, sbp, dbp in build_epoch_dataset(ecg_filtered, abp_raw, fs=FS_TARGET):
                feats = extract_features(ecg_epoch)
                feats["SBP"] = sbp
                feats["DBP"] = dbp
                feats["source_file"] = mat_path
                feats["record_index"] = record_idx
                rows.append(feats)

            n_processed += 1
            if n_processed % 50 == 0:
                logger.info("%d rekaman diproses, %d epoch terkumpul...", n_processed, len(rows))

    logger.info("Selesai: %d rekaman diproses (%d file), total %d epoch valid.", n_processed, len(mat_paths), len(rows))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangun dataset fitur dari sinyal ECG/ABP mentah (dataset asli).")
    parser.add_argument("--mat-path", default=None, help="Proses satu file .mat saja (override auto-discovery semua part_N.mat)")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.mat_path:
        mat_paths = [args.mat_path]
        logger.info("Mode satu file: %s", args.mat_path)
    else:
        mat_paths = discover_dataset_files()
        if not mat_paths:
            raise FileNotFoundError(f"Tidak ditemukan file part_N.mat di {DATASET_GLOB_PATTERN}")
        logger.info("Ditemukan %d file dataset: %s", len(mat_paths), [f.split('\\')[-1].split('/')[-1] for f in mat_paths])

    df = build_feature_dataframe(mat_paths, args.max_records)

    out_path = args.out or f"{PROCESSED_DATA_DIR}/features.csv"
    df.to_csv(out_path, index=False)
    logger.info("Dataset fitur disimpan ke %s (%d baris)", out_path, len(df))
