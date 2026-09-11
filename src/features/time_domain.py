"""
Ekstraksi 25 fitur statistik non-fiducial time-domain dari epoch ECG.

# Table 2, Kandaz & Ucar (2025) — "Representation of features mathematical and code"

extract_features() menerima epoch dengan panjang tetap EPOCH_LEN (1250 sampel
pada FS_TARGET=125Hz) dan mengembalikan dict 25 fitur sesuai FEATURE_NAMES di
src/config.py. Semua edge case (pembagian nol, nilai negatif untuk geometric/
harmonic mean) ditangani dengan fallback graceful + warning, tanpa crash.

CATATAN VERIFIKASI FORMULA (dicek ulang terhadap Table 2 jurnal, lihat komentar
per-fitur di bawah untuk detail):

- Table 2 baris #20 mendefinisikan S (dipakai di CV #4, Standard Error #19, dan
  Standard Deviation #20 sendiri) sebagai S = sqrt((1/n) * sum((x-x_bar)^2)) —
  yaitu STANDAR DEVIASI POPULASI (dibagi n), BUKAN standar deviasi sampel
  (dibagi n-1). Implementasi sebelumnya keliru memakai ddof=1 (sampel) —
  sudah diperbaiki di sini.
- Kurtosis (#1) & Skewness (#2) di Table 2 memakai S (populasi, lihat di atas)
  DITAMBAH faktor (n-1) terpisah di penyebut — bukan formula kurtosis/skewness
  populasi biasa. Diimplementasikan lewat faktor koreksi n/(n-1) dikalikan ke
  scipy.stats.kurtosis/skew versi populasi (turunan aljabar ada di bawah).
- Average Curve Length (#16) & Average Teager Energy (#25) di Table 2 eksplisit
  dibagi n (jumlah total sampel epoch), meskipun jumlah suku penjumlahannya
  n-1 dan n-2. np.mean() pada array hasil diff/teager akan membagi dengan
  jumlah SUKU (n-1 / n-2), bukan n — sudah diperbaiki jadi np.sum(...)/n.
- Singular Value Decomposition (#22) di Table 2 dituliskan sebagai `svd(x)`
  langsung pada vektor x. Di MATLAB, svd() pada vektor 1xN/Nx1 mengembalikan
  SATU nilai singular yang secara matematis sama dengan norma-2 (Euclidean)
  vektor tsb — svd([3 4]) == norm([3 4]) == 5. Implementasi sebelumnya keliru
  membangun trajectory/Hankel matrix (rekayasa fitur berbeda, bukan yang
  dimaksud formula `svd(x)` pada tabel) — sudah diperbaiki jadi norm(x).
- 25%/50% Trimmed Mean (#23, #24) memakai `trimmean(x, PERCENT)` gaya MATLAB,
  di mana PERCENT adalah persentase TOTAL yang dibuang (dibagi rata dua sisi:
  PERCENT/2 tiap sisi). scipy.stats.trim_mean(x, proportiontocut) sebaliknya
  memakai proportiontocut sebagai proporsi PER SISI. Jadi trimmean(x,25) harus
  dipetakan ke trim_mean(x, 0.125), dan trimmean(x,50) ke trim_mean(x, 0.25) —
  BUKAN trim_mean(x, 0.25) dan trim_mean(x, 0.5) seperti implementasi
  sebelumnya (yang terakhir bahkan selalu menghasilkan NaN untuk N genap,
  karena proportiontocut=0.5 membuang 100% data).
- Hjorth Mobility (#8) & Complexity (#9): rumus tercetak di Table 2 tampak
  kehilangan tanda akar (radikal) — kemungkinan besar artefak ekstraksi PDF
  dari tabel berbentuk gambar (pola serupa terlihat pada formula Complexity
  yang jadi tidak konsisten secara dimensi/aljabar bila dibaca literal tanpa
  akar). Hjorth Parameters adalah besaran baku dengan definisi tunggal yang
  sudah mapan sejak Hjorth (1970): Mobility = sqrt(var(x')/var(x)),
  Complexity = Mobility(x')/Mobility(x) — TETAP memakai definisi standar ini
  (sudah benar di versi sebelumnya, tidak diubah), bukan versi tanpa akar yang
  akan menghasilkan besaran dengan satuan/interpretasi berbeda dari namanya.
"""

import logging
import warnings

import numpy as np
from scipy import stats

from src.config import FEATURE_NAMES

logger = logging.getLogger(__name__)


def _hjorth_mobility(x: np.ndarray) -> float:
    """sqrt(var(x)/var(reference)) — std di sini SELALU populasi (ddof=0),
    konsisten dengan definisi S pada Table 2 baris #20."""
    std_x = np.std(x)
    if std_x == 0:
        return 0.0
    return float(np.std(np.diff(x)) / std_x)


def _svd_vector_norm(x: np.ndarray) -> float:
    """SVD(x) untuk x berupa vektor (Table 2 #22) == norma-2 Euclidean dari x.

    Ini identik dengan perilaku MATLAB svd(v) saat v adalah vektor 1xN/Nx1:
    satu-satunya nilai singular yang dihasilkan sama dengan norm(v).
    """
    return float(np.linalg.norm(x))


def extract_features(epoch: np.ndarray) -> dict:
    x = np.asarray(epoch, dtype=float)
    N = len(x)
    if N == 0:
        raise ValueError("epoch kosong, tidak bisa ekstraksi fitur")

    mean_x = float(np.mean(x))
    # Table 2 #20 — S = sqrt((1/n) * sum((x-x_bar)^2)) -> std POPULASI (ddof=0)
    std_x = float(np.std(x, ddof=0))
    median_x = float(np.median(x))

    # 3 — IQR
    q1, q3 = np.percentile(x, [25, 75])
    iqr = float(q3 - q1)

    # 4 — Coefficient of Variation: CV = (S/x_bar)*100, S = std populasi
    if mean_x == 0:
        logger.warning("mean=0, coefficient_of_variation fallback ke 0.0")
        cv = 0.0
    else:
        cv = float((std_x / mean_x) * 100)

    # 5 — Geometric Mean: ECG bisa negatif/nol -> shift ke positif sebelum gmean.
    # Table 2 mendefinisikan G=(x1*...*xn)^(1/n) langsung atas x tanpa membahas
    # kasus x negatif (padahal ECG terfilter tetap berosilasi di sekitar nol) —
    # gap ini kami tutup dengan shifting per-epoch supaya G selalu terdefinisi
    # real & finite, diterapkan identik pada semua epoch (training & real-time).
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

    # 7-9 — Hjorth Parameters (lihat catatan Mobility/Complexity di atas)
    hjorth_activity = float(np.var(x))  # A = S^2, populasi (ddof=0 default)
    hjorth_mobility = _hjorth_mobility(x)
    diff1 = np.diff(x)
    hjorth_mobility_diff1 = _hjorth_mobility(diff1) if len(diff1) > 1 else 0.0
    hjorth_complexity = (
        float(hjorth_mobility_diff1 / hjorth_mobility) if hjorth_mobility != 0 else 0.0
    )

    maximum = float(np.max(x))
    minimum = float(np.min(x))
    mad = float(np.mean(np.abs(x - mean_x)))

    # 14 — Central Moment orde-10 (scipy.stats.moment = populasi, sama seperti
    # MATLAB moment(x,order) yang tanpa koreksi bias — konsisten dengan tabel)
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        central_moment_10 = float(stats.moment(x, moment=10))
    if not np.isfinite(central_moment_10):
        logger.warning("central_moment_10 overflow/non-finite, fallback ke 0.0")
        central_moment_10 = 0.0

    # 16 — Average Curve Length: CL = (1/n) * sum_{i=2}^{n} |x_i - x_{i-1}|
    # Dibagi n (total sampel), BUKAN (n-1) jumlah suku diff.
    average_curve_length = float(np.sum(np.abs(np.diff(x))) / N) if N > 1 else 0.0

    average_energy = float(np.mean(x ** 2))
    rms = float(np.sqrt(average_energy))
    # 19 — Standard Error: S_xbar = S/sqrt(n), S = std populasi
    standard_error = float(std_x / np.sqrt(N)) if N > 0 else 0.0

    mean_abs_x = float(np.mean(np.abs(x)))
    if mean_abs_x == 0:
        logger.warning("mean(|x|)=0, shape_factor fallback ke 0.0")
        shape_factor = 0.0
    else:
        shape_factor = float(rms / mean_abs_x)

    # 22 — SVD(x) untuk vektor x == norma-2 Euclidean (lihat _svd_vector_norm)
    svd_dominant = _svd_vector_norm(x)

    # 23-24 — Trimmed Mean gaya MATLAB trimmean(x, percent): percent = total
    # persentase dibuang (dibagi dua sisi). trimmean(x,25) -> proportiontocut
    # 0.125/sisi; trimmean(x,50) -> proportiontocut 0.25/sisi.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_trim25 = stats.trim_mean(x, 0.125) if N > 0 else np.nan
        raw_trim50 = stats.trim_mean(x, 0.25) if N > 0 else np.nan
    trimmed_mean_25 = float(raw_trim25) if np.isfinite(raw_trim25) else median_x
    trimmed_mean_50 = float(raw_trim50) if np.isfinite(raw_trim50) else median_x

    # 25 — Average Teager Energy: TE = (1/n) * sum_{i=3}^{n} (x_{i-1}^2 - x_i*x_{i-2})
    # Dibagi n (total sampel), BUKAN (n-2) jumlah suku.
    if N > 2:
        teager = x[1:-1] ** 2 - x[:-2] * x[2:]
        average_teager_energy = float(np.sum(teager) / N)
    else:
        logger.warning("epoch terlalu pendek untuk Teager Energy, fallback ke 0.0")
        average_teager_energy = 0.0

    # 1-2 — Kurtosis & Skewness: Table 2 memakai S (populasi, lihat #20) dengan
    # faktor tambahan (n-1) di penyebut:
    #   x_kur = sum((x-x_bar)^4) / ((n-1)*S^4)
    #   x_ske = sum((x-x_bar)^3) / ((n-1)*S^3)
    # Menulis M2=var populasi, M3/M4=momen populasi orde-3/4 (=scipy skew/
    # kurtosis populasi baku), aljabar ini persis sama dengan
    # (n/(n-1)) * scipy.stats.{skew,kurtosis}(x, ...) versi populasi -
    # dipakai di sini alih-alih formula kurtosis/skewness populasi biasa.
    n_correction = N / (N - 1) if N > 1 else 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_kurtosis = stats.kurtosis(x, fisher=False) * n_correction
        raw_skewness = stats.skew(x) * n_correction
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
