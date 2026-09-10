"""
Ekstraksi 25 fitur statistik non-fiducial time-domain dari epoch ECG.

# Bagian 3.x proposal — Ekstraksi fitur non-fiducial (Kandaz & Ucar, 2025)

extract_features() menerima epoch dengan panjang tetap EPOCH_LEN (1250 sampel
pada FS_TARGET=125Hz) dan mengembalikan dict 25 fitur sesuai FEATURE_NAMES di
src/config.py. Semua edge case (pembagian nol, nilai negatif untuk geometric/
harmonic mean) ditangani dengan fallback graceful + warning, tanpa crash.
"""

import logging
import warnings

import numpy as np
from scipy import stats
from scipy.linalg import hankel

from src.config import FEATURE_NAMES

logger = logging.getLogger(__name__)


def _hjorth_mobility(x: np.ndarray) -> float:
    std_x = np.std(x)
    if std_x == 0:
        return 0.0
    return float(np.std(np.diff(x)) / std_x)


def _svd_dominant(x: np.ndarray, embed_dim: int = 10) -> float:
    """Nilai singular dominan dari trajectory matrix (delay embedding).

    Trajectory matrix dibentuk lewat scipy.linalg.hankel: setiap baris adalah
    window bergeser sepanjang embed_dim sampel dari sinyal x. SVD dari matrix
    ini mendekomposisi sinyal ke mode-mode ortogonal; nilai singular pertama
    (terbesar) merepresentasikan energi mode dominan sinyal.
    """
    n = len(x)
    if n < embed_dim:
        embed_dim = max(1, n)
    col = x[: n - embed_dim + 1]
    row = x[n - embed_dim :]
    traj_matrix = hankel(col, row)
    try:
        singular_values = np.linalg.svd(traj_matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        logger.warning("SVD gagal konvergen, fallback ke 0.0")
        return 0.0
    return float(singular_values[0]) if singular_values.size else 0.0


def extract_features(epoch: np.ndarray) -> dict:
    x = np.asarray(epoch, dtype=float)
    N = len(x)
    if N == 0:
        raise ValueError("epoch kosong, tidak bisa ekstraksi fitur")

    mean_x = float(np.mean(x))
    std_x = float(np.std(x, ddof=1)) if N > 1 else 0.0
    median_x = float(np.median(x))

    # 3 — IQR
    q1, q3 = np.percentile(x, [25, 75])
    iqr = float(q3 - q1)

    # 4 — Coefficient of Variation
    if mean_x == 0:
        logger.warning("mean=0, coefficient_of_variation fallback ke 0.0")
        cv = 0.0
    else:
        cv = float((std_x / mean_x) * 100)

    # 5 — Geometric Mean: ECG bisa negatif -> shift ke positif sebelum gmean
    shift = abs(np.min(x)) + 1e-9 if np.min(x) <= 0 else 0.0
    x_pos = x + shift
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geometric_mean = float(stats.gmean(x_pos))

    # 6 — Harmonic Mean: tangani nilai nol dengan shifting yang sama
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_pos_nonzero = np.where(x_pos == 0, 1e-9, x_pos)
        harmonic_mean = float(stats.hmean(x_pos_nonzero))

    # 7-9 — Hjorth Parameters
    hjorth_activity = float(np.var(x))
    hjorth_mobility = _hjorth_mobility(x)
    diff1 = np.diff(x)
    hjorth_mobility_diff1 = _hjorth_mobility(diff1) if len(diff1) > 1 else 0.0
    hjorth_complexity = (
        float(hjorth_mobility_diff1 / hjorth_mobility) if hjorth_mobility != 0 else 0.0
    )

    maximum = float(np.max(x))
    minimum = float(np.min(x))
    mad = float(np.mean(np.abs(x - mean_x)))

    # 14 — Central Moment orde-10 (rawan overflow untuk sinyal beramplitudo besar)
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        central_moment_10 = float(stats.moment(x, moment=10))
    if not np.isfinite(central_moment_10):
        logger.warning("central_moment_10 overflow/non-finite, fallback ke 0.0")
        central_moment_10 = 0.0

    average_curve_length = float(np.mean(np.abs(np.diff(x)))) if N > 1 else 0.0
    average_energy = float(np.mean(x ** 2))
    rms = float(np.sqrt(average_energy))
    standard_error = float(std_x / np.sqrt(N)) if N > 0 else 0.0

    mean_abs_x = float(np.mean(np.abs(x)))
    if mean_abs_x == 0:
        logger.warning("mean(|x|)=0, shape_factor fallback ke 0.0")
        shape_factor = 0.0
    else:
        shape_factor = float(rms / mean_abs_x)

    svd_dominant = _svd_dominant(x)

    trimmed_mean_25 = float(stats.trim_mean(x, 0.25))
    # trim_mean(x, 0.5) cuts 50% from BOTH tails; for even N this leaves an
    # empty slice (NaN). The table defines this feature as "~ median", so we
    # fall back to the median whenever the trim degenerates to nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_trim50 = stats.trim_mean(x, 0.5) if N > 0 else np.nan
    if not np.isfinite(raw_trim50):
        trimmed_mean_50 = median_x
    else:
        trimmed_mean_50 = float(raw_trim50)

    # 25 — Average Teager Energy
    if N > 2:
        teager = x[1:-1] ** 2 - x[:-2] * x[2:]
        average_teager_energy = float(np.mean(teager))
    else:
        logger.warning("epoch terlalu pendek untuk Teager Energy, fallback ke 0.0")
        average_teager_energy = 0.0

    # kurtosis/skewness are 0/0 (NaN) when the signal has zero variance
    # (e.g. a flat epoch); conventionally reported as 0.0 in that case.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_kurtosis = stats.kurtosis(x, fisher=False)
        raw_skewness = stats.skew(x)
    kurtosis_val = float(raw_kurtosis) if np.isfinite(raw_kurtosis) else 0.0
    skewness_val = float(raw_skewness) if np.isfinite(raw_skewness) else 0.0
    if not np.isfinite(raw_kurtosis) or not np.isfinite(raw_skewness):
        logger.warning("kurtosis/skewness non-finite (variansi nol), fallback ke 0.0")

    features = {
        "kurtosis": kurtosis_val,
        "skewness": skewness_val,
        "iqr": iqr,
        "coefficient_of_variation": cv,
        "geometric_mean": geometric_mean,
        "harmonic_mean": harmonic_mean,
        "hjorth_activity": hjorth_activity,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
        "maximum": maximum,
        "median": median_x,
        "mean_absolute_deviation": mad,
        "minimum": minimum,
        "central_moment_10": central_moment_10,
        "mean": mean_x,
        "average_curve_length": average_curve_length,
        "average_energy": average_energy,
        "root_mean_square": rms,
        "standard_error": standard_error,
        "standard_deviation": std_x,
        "shape_factor": shape_factor,
        "svd_dominant": svd_dominant,
        "trimmed_mean_25": trimmed_mean_25,
        "trimmed_mean_50": trimmed_mean_50,
        "average_teager_energy": average_teager_energy,
    }

    assert list(features.keys()) == FEATURE_NAMES, "urutan fitur tidak sinkron dengan config.FEATURE_NAMES"
    return features
