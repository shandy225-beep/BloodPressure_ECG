"""
Verifikasi rumus 25 fitur time-domain di src/features/time_domain.py.

src/features/time_domain.py SENGAJA ditulis TANPA numpy/scipy: setiap rumus
adalah aritmetika dasar (loop Python + modul `math`), supaya bisa ditelusuri
baris-per-baris dan dihitung ulang di kertas -- tidak ada fungsi statistik
"kotak hitam" (np.std, scipy.stats.kurtosis, dst.) yang hasilnya harus
dipercaya begitu saja. Lihat docstring di file itu untuk detail tiap rumus.

Skrip ini membuktikan kebenarannya dengan DUA cara independen:
  1) Uji hitung-tangan: array kecil [2,4,4,4,5,5,7,9] yang bisa dihitung di
     kertas (mean, std, kurtosis, dst. sebagai pecahan eksak), dibandingkan
     dengan keluaran extract_features() (kode PRODUKSI, pure Python).
  2) Uji silang: extract_features() (pure Python, PRODUKSI) dibandingkan
     dengan reference_features_numpy() di bawah -- rumus yang SAMA dihitung
     lewat numpy/scipy sebagai pembanding INDEPENDEN. numpy/scipy di sini
     HANYA dipakai untuk validasi di skrip ini, TIDAK dipakai lagi di
     pipeline produksi.

Tujuan: membuktikan rumus di time_domain.py setara dengan Table 2
Kandaz & Ucar (2025) dan Bab 2.2.6 proposal, TANPA bergantung pada
kebenaran library statistik mana pun untuk pembuktiannya sendiri.

Jalankan dari root proyek:  python verifikasi_rumus_manual.py
(Sesuaikan baris import extract_features di bawah dengan lokasi file Anda.)
"""
import math
import warnings

import numpy as np       # HANYA dipakai di skrip verifikasi ini (referensi independen)
from scipy import stats  # HANYA dipakai di skrip verifikasi ini (referensi independen)

try:
    from src.features.time_domain import extract_features   # kode PRODUKSI (pure Python)
except ImportError:
    from time_domain import extract_features


# ------------------------------------------------- referensi independen (numpy/scipy)
def reference_features_numpy(x):
    """Rumus yang SAMA persis, dihitung lewat numpy/scipy sebagai pembanding
    independen -- dipakai HANYA untuk validasi di skrip ini. Pipeline produksi
    (src/features/time_domain.py) TIDAK memanggil fungsi ini maupun numpy/scipy."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = float(np.mean(x))
    S = float(np.std(x, ddof=0))
    xs = np.sort(x)
    f = {}

    f["mean"] = mu
    f["standard_deviation"] = S
    f["median"] = float(np.median(x))
    q1, q3 = np.percentile(x, [25, 75])
    f["iqr"] = float(q3 - q1)
    f["coefficient_of_variation"] = float((S / mu) * 100) if mu != 0 else 0.0

    shift = abs(float(xs[0])) + 1e-9 if xs[0] <= 0 else 0.0
    x_pos = x + shift
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f["geometric_mean"] = float(stats.gmean(x_pos))
        x_pos_nonzero = np.where(x_pos == 0, 1e-9, x_pos)
        f["harmonic_mean"] = float(stats.hmean(x_pos_nonzero))

    f["hjorth_activity"] = float(np.var(x))

    def _mobility(v):
        s = np.std(v)
        return float(np.std(np.diff(v)) / s) if s != 0 else 0.0

    f["hjorth_mobility"] = _mobility(x)
    d1 = np.diff(x)
    mob1 = _mobility(d1) if len(d1) > 1 else 0.0
    f["hjorth_complexity"] = float(mob1 / f["hjorth_mobility"]) if f["hjorth_mobility"] != 0 else 0.0

    f["maximum"] = float(xs[-1])
    f["minimum"] = float(xs[0])
    f["mean_absolute_deviation"] = float(np.mean(np.abs(x - mu)))

    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm10 = float(stats.moment(x, moment=10))
    f["central_moment_10"] = cm10 if np.isfinite(cm10) else 0.0

    f["average_curve_length"] = float(np.sum(np.abs(np.diff(x))) / n) if n > 1 else 0.0
    E = float(np.mean(x ** 2))
    f["average_energy"] = E
    f["root_mean_square"] = math.sqrt(E)
    f["standard_error"] = S / math.sqrt(n)

    mean_abs = float(np.mean(np.abs(x)))
    f["shape_factor"] = float(f["root_mean_square"] / mean_abs) if mean_abs != 0 else 0.0
    f["svd_dominant"] = float(np.linalg.norm(x))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t25, t50 = stats.trim_mean(x, 0.125), stats.trim_mean(x, 0.25)
    f["trimmed_mean_25"] = float(t25) if np.isfinite(t25) else f["median"]
    f["trimmed_mean_50"] = float(t50) if np.isfinite(t50) else f["median"]

    if n > 2:
        teager = x[1:-1] ** 2 - x[:-2] * x[2:]
        f["average_teager_energy"] = float(np.sum(teager) / n)
    else:
        f["average_teager_energy"] = 0.0

    nc = n / (n - 1) if n > 1 else 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kur = stats.kurtosis(x, fisher=False) * nc
        ske = stats.skew(x) * nc
    f["kurtosis"] = float(kur) if np.isfinite(kur) else 0.0
    f["skewness"] = float(ske) if np.isfinite(ske) else 0.0

    return f


# ------------------------------------------ 1) uji hitung-tangan (bisa diverifikasi di kertas)
def test_hand_calculation():
    x = [2, 4, 4, 4, 5, 5, 7, 9]      # contoh klasik: mean=5, std populasi=2
    f = extract_features(np.array(x, dtype=float))
    expected = {
        "mean": 5.0,
        "standard_deviation": 2.0,           # sqrt(32/8)
        "hjorth_activity": 4.0,              # 32/8
        "kurtosis": 356 / (7 * 16),          # sum d^4 = 356, (n-1) S^4 = 7*16
        "skewness": 42 / (7 * 8),            # sum d^3 = 42,  (n-1) S^3 = 7*8
        "coefficient_of_variation": 40.0,    # 2/5*100
        "average_energy": 29.0,              # 232/8
        "root_mean_square": math.sqrt(29.0),
        "median": 4.5,
        "maximum": 9.0,
        "minimum": 2.0,
        "iqr": 1.5,                          # Q1=4, Q3=5.5
        "average_curve_length": 7 / 8,       # sum|diff| = 7
        "average_teager_energy": 3 / 8,      # sum = 8+0-4+5-10+4 = 3
        "standard_error": 2.0 / math.sqrt(8),
        "svd_dominant": math.sqrt(232),
        "mean_absolute_deviation": 12 / 8,   # |d| = 3,1,1,1,0,0,2,4 -> 12
    }
    ok = True
    for k, v in expected.items():
        good = math.isclose(f[k], v, rel_tol=1e-9)
        ok &= good
        print(f"  [{'OK' if good else 'GAGAL'}] {k:26s} tangan={v:.6f}  kode={f[k]:.6f}")
    return ok


# ------------------------------------------ 2) uji silang: produksi (pure Python) vs numpy/scipy
def make_epoch(seed, n=1250):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 125.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * t) + 0.02 * rng.standard_normal(n) - 0.05
    for k in range(8):
        x[40 + 150 * k] += 1.0                 # R-peak sintetis
    return x


def test_cross_check(n_epochs=5, rtol=1e-6):
    ok = True
    worst = {}
    for seed in range(n_epochs):
        x = make_epoch(seed)
        prod = extract_features(x)              # PRODUKSI: pure Python (time_domain.py)
        ref = reference_features_numpy(x)        # REFERENSI independen: numpy/scipy
        for k, v in ref.items():
            err = abs(prod[k] - v) / (abs(v) + 1e-300)
            worst[k] = max(worst.get(k, 0.0), err)
    print(f"  {'fitur':28s} {'galat relatif maks':>20s}")
    for k, e in worst.items():
        good = e < rtol
        ok &= good
        print(f"  [{'OK' if good else 'GAGAL'}] {k:26s} {e:18.2e}")
    return ok


if __name__ == "__main__":
    print("== 1) Uji hitung-tangan (x = [2,4,4,4,5,5,7,9]) ==")
    a = test_hand_calculation()
    print("\n== 2) Uji silang: extract_features() PRODUKSI (pure Python) vs numpy/scipy (5 epoch) ==")
    b = test_cross_check()
    print("\nHASIL:", "SEMUA SESUAI" if (a and b) else "ADA YANG TIDAK SESUAI")
