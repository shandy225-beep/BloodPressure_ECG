"""
Loader untuk dataset Kaggle "Cuff-Less Blood Pressure Estimation"
(mkachuee/BloodPressureDataset), format part_N.mat.

Struktur .mat: key 'p' berisi array object, satu elemen per rekaman pasien.
Setiap elemen berbentuk (3, n_samples): baris 0 = PPG, baris 1 = ABP,
baris 2 = ECG — sudah native 125 Hz (FS_TARGET), TIDAK perlu resampling.
"""

import logging
import os

import numpy as np
import scipy.io

logger = logging.getLogger(__name__)

PPG_ROW, ABP_ROW, ECG_ROW = 0, 1, 2


def load_mat_records(path: str):
    """Generator yang menghasilkan (ecg, abp) per rekaman pasien dari file .mat.

    ecg, abp: np.ndarray 1D, native 125 Hz (FS_TARGET), tidak di-resample.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {path}. Unduh 'Cuff-Less Blood Pressure "
            f"Estimation' dari Kaggle (mkachuee/BloodPressureDataset), lalu taruh "
            f"part_1.mat ... part_12.mat di folder Dataset/. Lihat README untuk instruksi lengkap."
        )

    logger.info("Memuat dataset dari %s ...", path)
    mat_data = scipy.io.loadmat(path)
    records = mat_data["p"][0]
    logger.info("Ditemukan %d rekaman pasien.", len(records))

    for record in records:
        ecg = np.asarray(record[ECG_ROW, :], dtype=float)
        abp = np.asarray(record[ABP_ROW, :], dtype=float)
        yield ecg, abp
