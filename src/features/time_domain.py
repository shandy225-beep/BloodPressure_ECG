"""
Ekstraksi 25 fitur statistik non-fiducial time-domain dari epoch ECG.

# Table 2, Kandaz & Ucar (2025) — "Representation of features mathematical and code"

extract_features() menerima epoch dengan panjang tetap EPOCH_LEN (1250 sampel
pada FS_TARGET=125Hz) dan mengembalikan dict 25 fitur sesuai FEATURE_NAMES di
src/config.py. Semua edge case (pembagian nol, nilai negatif untuk geometric/
harmonic mean) ditangani dengan fallback graceful + warning, tanpa crash.

IMPLEMENTASI SENGAJA TIDAK MEMAKAI numpy/scipy: setiap rumus ditulis sebagai
aritmetika dasar (`+ - * /`, `math.sqrt`, `math.log`) atas list Python biasa,
persis seperti rumus di Table 2 -- tidak ada fungsi statistik "kotak hitam"
(np.std, scipy.stats.kurtosis, dst.) yang hasilnya harus dipercaya begitu
saja. Ini supaya tiap rumus bisa ditelusuri baris-per-baris dan dihitung
ulang di kertas. Konsekuensinya, numpy TIDAK memberi jaring pengaman lagi:
Python murni MELEMPAR ZeroDivisionError untuk x/0.0 dan OverflowError untuk
pangkat yang meluap (numpy diam-diam menghasilkan inf/nan) -- setiap titik
rawan itu dijaga eksplisit di bawah (lihat try/except & pengecekan ==0).

Verifikasi kesetaraan dengan rumus & dengan numpy/scipy sebagai referensi
independen ada di verifikasi_rumus_manual.py (root proyek): uji hitung-tangan
(bisa dicek di kertas) + uji silang terhadap numpy/scipy pada epoch realistik.

CATATAN VERIFIKASI FORMULA (dicek ulang terhadap Table 2 jurnal, lihat komentar
per-fitur di bawah untuk detail):

- Table 2 baris #20 mendefinisikan S (dipakai di CV #4, Standard Error #19, dan
  Standard Deviation #20 sendiri) sebagai S = sqrt((1/n) * sum((x-x_bar)^2)) —
  yaitu STANDAR DEVIASI POPULASI (dibagi n), BUKAN standar deviasi sampel
  (dibagi n-1).
- Kurtosis (#1) & Skewness (#2) di Table 2 memakai S (populasi, lihat di atas)
  DITAMBAH faktor (n-1) terpisah di penyebut — bukan formula kurtosis/skewness
  populasi biasa: x_kur = sum((x-x_bar)^4)/((n-1)*S^4), begitu juga skewness
  dengan pangkat 3. Ditulis LANGSUNG sesuai formula ini (bukan lewat turunan
  aljabar dari fungsi library manapun).
- Average Curve Length (#16) & Average Teager Energy (#25) di Table 2 eksplisit
  dibagi n (jumlah total sampel epoch), meskipun jumlah suku penjumlahannya
  n-1 dan n-2.
- Singular Value Decomposition (#22) di Table 2 dituliskan sebagai `svd(x)`
  langsung pada vektor x. Di MATLAB, svd() pada vektor 1xN/Nx1 mengembalikan
  SATU nilai singular yang secara matematis sama dengan norma-2 (Euclidean)
  vektor tsb — svd([3 4]) == norm([3 4]) == 5, diimplementasikan sebagai
  sqrt(sum(x_i^2)).
- 25%/50% Trimmed Mean (#23, #24) memakai `trimmean(x, PERCENT)` gaya MATLAB,
  di mana PERCENT adalah persentase TOTAL yang dibuang (dibagi rata dua sisi:
  PERCENT/2 tiap sisi): buang floor((PERCENT/2/100... sudah dalam proporsi)*n)
  elemen terkecil & terbesar (hasil sort), lalu rata-ratakan sisanya.
  trimmean(x,25) -> buang 12.5% tiap sisi; trimmean(x,50) -> buang 25% tiap sisi.
- Hjorth Mobility (#8) & Complexity (#9): rumus tercetak di Table 2 tampak
  kehilangan tanda akar (radikal) — kemungkinan besar artefak ekstraksi PDF
  dari tabel berbentuk gambar (pola serupa terlihat pada formula Complexity
  yang jadi tidak konsisten secara dimensi/aljabar bila dibaca literal tanpa
  akar). Hjorth Parameters adalah besaran baku dengan definisi tunggal yang
  sudah mapan sejak Hjorth (1970): Mobility = sqrt(var(x')/var(x)),
  Complexity = Mobility(x')/Mobility(x) — TETAP memakai definisi standar ini,
  bukan versi tanpa akar yang akan menghasilkan besaran dengan
  satuan/interpretasi berbeda dari namanya.
"""

import logging
import math

from src.config import FEATURE_NAMES

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Fungsi bantu aritmetika dasar (pure Python, tanpa numpy/scipy)
# --------------------------------------------------------------------------
def _mean(x):
    return sum(x) / len(x)


def _std_pop(x, mu=None):
    """Standar deviasi POPULASI: S = sqrt((1/n) * sum((x-x_bar)^2)) -- Table 2 #20."""
    if mu is None:
        mu = _mean(x)
    return math.sqrt(sum((v - mu) ** 2 for v in x) / len(x))


def _diff(x):
    """Selisih berurutan: [x1-x0, x2-x1, ...] -- setara np.diff()."""
    return [x[i] - x[i - 1] for i in range(1, len(x))]


def _quantile_linear(xs_sorted, q):
    """Kuantil interpolasi linear (Hyndman-Fan tipe 7 = default numpy.percentile).
    xs_sorted HARUS sudah terurut naik."""
    n = len(xs_sorted)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return xs_sorted[lo] + (pos - lo) * (xs_sorted[hi] - xs_sorted[lo])


def _trim_mean(xs_sorted, proportion_per_side):
    """Buang floor(proportion_per_side * n) elemen terkecil & terbesar dari
    xs_sorted (HARUS sudah terurut naik), lalu rata-ratakan sisanya -- setara
    scipy.stats.trim_mean(x, proportiontocut)."""
    n = len(xs_sorted)
    k = int(proportion_per_side * n)
    core = xs_sorted[k: n - k]
    return sum(core) / len(core)


def _hjorth_mobility(x):
    """Hjorth Mobility = std(x') / std(x) = sqrt(var(x')/var(x)), x' = diff(x).
    std di sini SELALU populasi (ddof=0), konsisten dengan definisi S pada
    Table 2 baris #20. Complexity memanggil fungsi ini pada x dan pada x'.

    Sinyal konstan (std(x)=0) mengembalikan 0.0 langsung -- kalau tidak,
    Python murni akan melempar ZeroDivisionError pada std(x')/0.0 (numpy diam-
    diam menghasilkan nan/inf untuk kasus ini, jadi guard ini WAJIB di versi
    pure-Python)."""
    std_x = _std_pop(x)
    if std_x == 0:
        return 0.0
    return _std_pop(_diff(x)) / std_x


def _svd_vector_norm(x):
    """SVD(x) untuk x berupa vektor (Table 2 #22) == norma-2 Euclidean dari x.

    Ini identik dengan perilaku MATLAB svd(v) saat v adalah vektor 1xN/Nx1:
    satu-satunya nilai singular yang dihasilkan sama dengan norm(v).
    """
    return math.sqrt(sum(v * v for v in x))


def extract_features(epoch) -> dict:
    """epoch: sequence numerik apa pun (list, tuple, atau numpy.ndarray 1D)."""
    x = [float(v) for v in epoch]
    N = len(x)
    if N == 0:
        raise ValueError("epoch kosong, tidak bisa ekstraksi fitur")

    mean_x = _mean(x)
    # Table 2 #20 — S = sqrt((1/n) * sum((x-x_bar)^2)) -> std POPULASI (ddof=0)
    std_x = _std_pop(x, mu=mean_x)
    xs = sorted(x)          # dipakai ulang: median, IQR, min/max, trimmed mean
    median_x = xs[N // 2] if N % 2 else 0.5 * (xs[N // 2 - 1] + xs[N // 2])

    # 3 — IQR (kuantil interpolasi linear, sama seperti np.percentile default)
    q1 = _quantile_linear(xs, 0.25)
    q3 = _quantile_linear(xs, 0.75)
    iqr = q3 - q1

    # 4 — Coefficient of Variation: CV = (S/x_bar)*100, S = std populasi
    if mean_x == 0:
        logger.warning("mean=0, coefficient_of_variation fallback ke 0.0")
        cv = 0.0
    else:
        cv = (std_x / mean_x) * 100

    # 5 — Geometric Mean: ECG bisa negatif/nol -> shift ke positif sebelum gmean.
    # Table 2 mendefinisikan G=(x1*...*xn)^(1/n) langsung atas x tanpa membahas
    # kasus x negatif (padahal ECG terfilter tetap berosilasi di sekitar nol) —
    # gap ini kami tutup dengan shifting per-epoch supaya G selalu terdefinisi
    # real & finite, diterapkan identik pada semua epoch (training & real-time).
    # Shift ini menjamin semua nilai > 0 (min baru = 1e-9), jadi math.log() di
    # bawah tidak pernah menerima 0/negatif.
    shift = abs(xs[0]) + 1e-9 if xs[0] <= 0 else 0.0
    x_pos = [v + shift for v in x]
    geometric_mean = math.exp(sum(math.log(v) for v in x_pos) / N)   # = (prod x)^(1/n)

    # 6 — Harmonic Mean: tangani nilai nol dengan shifting yang sama
    x_pos_nonzero = [v if v != 0 else 1e-9 for v in x_pos]
    harmonic_mean = N / sum(1.0 / v for v in x_pos_nonzero)

    # 7-9 — Hjorth Parameters (lihat catatan Mobility/Complexity di atas)
    hjorth_activity = std_x ** 2  # A = S^2, populasi
    hjorth_mobility = _hjorth_mobility(x)
    diff1 = _diff(x)
    hjorth_mobility_diff1 = _hjorth_mobility(diff1) if len(diff1) > 1 else 0.0
    hjorth_complexity = (hjorth_mobility_diff1 / hjorth_mobility) if hjorth_mobility != 0 else 0.0

    maximum = xs[-1]
    minimum = xs[0]
    mad = sum(abs(v - mean_x) for v in x) / N

    # 14 — Central Moment orde-10 (populasi, tanpa koreksi bias -- konsisten
    # dengan MATLAB moment(x,order) dan Table 2). Pangkat 10 bisa meluap
    # (OverflowError di Python murni, beda dgn numpy yg diam2 jadi inf) untuk
    # nilai ekstrem -- ditangkap & di-fallback sama seperti nilai non-finite.
    try:
        central_moment_10 = sum((v - mean_x) ** 10 for v in x) / N
    except OverflowError:
        central_moment_10 = math.inf
    if not math.isfinite(central_moment_10):
        logger.warning("central_moment_10 overflow/non-finite, fallback ke 0.0")
        central_moment_10 = 0.0

    # 16 — Average Curve Length: CL = (1/n) * sum_{i=2}^{n} |x_i - x_{i-1}|
    # Dibagi n (total sampel), BUKAN (n-1) jumlah suku diff.
    average_curve_length = sum(abs(x[i] - x[i - 1]) for i in range(1, N)) / N if N > 1 else 0.0

    average_energy = sum(v * v for v in x) / N
    rms = math.sqrt(average_energy)
    # 19 — Standard Error: S_xbar = S/sqrt(n), S = std populasi
    standard_error = std_x / math.sqrt(N)

    mean_abs_x = sum(abs(v) for v in x) / N
    if mean_abs_x == 0:
        logger.warning("mean(|x|)=0, shape_factor fallback ke 0.0")
        shape_factor = 0.0
    else:
        shape_factor = rms / mean_abs_x

    # 22 — SVD(x) untuk vektor x == norma-2 Euclidean (lihat _svd_vector_norm)
    svd_dominant = _svd_vector_norm(x)

    # 23-24 — Trimmed Mean gaya MATLAB trimmean(x, percent): percent = total
    # persentase dibuang (dibagi dua sisi). trimmean(x,25) -> proportiontocut
    # 0.125/sisi; trimmean(x,50) -> proportiontocut 0.25/sisi. Untuk proporsi
    # <=0.25/sisi, sisa data (core) tidak pernah kosong untuk n>=1, jadi
    # fallback ke median di sini murni jaring pengaman, bukan jalur yang
    # tereksekusi pada penggunaan normal.
    raw_trim25 = _trim_mean(xs, 0.125)
    raw_trim50 = _trim_mean(xs, 0.25)
    trimmed_mean_25 = raw_trim25 if math.isfinite(raw_trim25) else median_x
    trimmed_mean_50 = raw_trim50 if math.isfinite(raw_trim50) else median_x

    # 25 — Average Teager Energy: TE = (1/n) * sum_{i=3}^{n} (x_{i-1}^2 - x_i*x_{i-2})
    # Dibagi n (total sampel), BUKAN (n-2) jumlah suku.
    if N > 2:
        average_teager_energy = sum(x[i - 1] ** 2 - x[i] * x[i - 2] for i in range(2, N)) / N
    else:
        logger.warning("epoch terlalu pendek untuk Teager Energy, fallback ke 0.0")
        average_teager_energy = 0.0

    # 1-2 — Kurtosis & Skewness: Table 2 memakai S (populasi, lihat #20) dengan
    # faktor tambahan (n-1) di penyebut, ditulis LANGSUNG sesuai formula:
    #   x_kur = sum((x-x_bar)^4) / ((n-1)*S^4)
    #   x_ske = sum((x-x_bar)^3) / ((n-1)*S^3)
    # (n-1)==0 (N=1) atau S==0 (sinyal konstan) membuat penyebut persis nol --
    # Python murni akan ZeroDivisionError di sini (numpy diam-diam jadi nan),
    # jadi guard eksplisit ke NaN dulu, baru fallback lewat isfinite di bawah,
    # supaya perilakunya SETARA numpy tanpa numpy.
    if N > 1 and std_x != 0:
        try:
            raw_kurtosis = sum((v - mean_x) ** 4 for v in x) / ((N - 1) * std_x ** 4)
            raw_skewness = sum((v - mean_x) ** 3 for v in x) / ((N - 1) * std_x ** 3)
        except OverflowError:
            raw_kurtosis = raw_skewness = math.nan
    else:
        raw_kurtosis = raw_skewness = math.nan
    kurtosis_val = raw_kurtosis if math.isfinite(raw_kurtosis) else 0.0
    skewness_val = raw_skewness if math.isfinite(raw_skewness) else 0.0
    if not math.isfinite(raw_kurtosis) or not math.isfinite(raw_skewness):
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
