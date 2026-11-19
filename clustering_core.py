"""
clustering_core.py
==================
Modul inti untuk pengelompokan (clustering) alamat siswa antar jemput sekolah
dengan batasan kapasitas kendaraan (Capacitated Clustering).

Ketentuan:
- Satu cluster = satu kendaraan antar jemput.
- Satu kendaraan mengangkut maksimal `kapasitas` siswa (default 10).
- Tujuan: alamat yang berdekatan berada dalam satu cluster.

Metode yang tersedia:
1. "kmeans" : Capacitated K-Means (K-Means + penugasan seimbang dengan
              algoritma Hungarian / linear_sum_assignment).
2. "sweep"  : Sweep Algorithm (alamat diurutkan menurut sudut terhadap
              sekolah, lalu diisi sampai kapasitas penuh).
3. "auto"   : Menjalankan keduanya dan memilih cluster yang paling kompak.

Hasil awal disempurnakan dengan local search (pindah/tukar alamat
antar-cluster) agar setiap cluster serapat mungkin.
Program ini TIDAK menghitung urutan jemput maupun jarak tempuh, karena
kondisi jalan sangat bervariasi; urutan rute diserahkan kepada pengemudi.

Dependensi: numpy, pandas, scikit-learn, scipy
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

R_BUMI_KM = 6371.0088

# Koordinat sekolah bawaan (dapat diubah di aplikasi / notebook / skrip)
SEKOLAH_DEFAULT = (-1.634472, 103.492810)

# Nama kolom yang dikenali (huruf kecil). Kolom pertama yang ditemukan dipakai.
KANDIDAT_KOLOM = {
    "alamat": ["alamat", "address", "nama_alamat", "lokasi"],
    "lat": ["x", "lat", "latitude", "lintang"],
    "lon": ["y", "lon", "lng", "long", "longitude", "bujur"],
}


# ---------------------------------------------------------------------------
# 1. Membaca & memvalidasi data
# ---------------------------------------------------------------------------
def deteksi_kolom(df: pd.DataFrame) -> dict:
    """Mencocokkan nama kolom pada file dengan nama standar."""
    peta = {}
    kolom_lower = {str(c).strip().lower(): c for c in df.columns}
    for standar, kandidat in KANDIDAT_KOLOM.items():
        for k in kandidat:
            if k in kolom_lower:
                peta[standar] = kolom_lower[k]
                break
    return peta


def siapkan_data(df_mentah: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    """
    Membersihkan data dan mengembalikan (df_bersih, peta_kolom, peringatan).
    df_bersih memiliki kolom: id_siswa, alamat, lat, lon. Kolom lain diabaikan.
    """
    peringatan = []
    peta = deteksi_kolom(df_mentah)
    hilang = [w for w in ["alamat", "lat", "lon"] if w not in peta]
    if hilang:
        raise ValueError(
            "Kolom wajib tidak ditemukan: " + ", ".join(hilang)
            + ". Pastikan file memiliki kolom: alamat, x (latitude), y (longitude)."
        )

    df = pd.DataFrame()
    df["id_siswa"] = np.arange(1, len(df_mentah) + 1)
    df["alamat"] = df_mentah[peta["alamat"]].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df_mentah[peta["lat"]], errors="coerce")
    df["lon"] = pd.to_numeric(df_mentah[peta["lon"]], errors="coerce")

    n_awal = len(df)
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if len(df) < n_awal:
        peringatan.append(f"{n_awal - len(df)} baris dibuang karena koordinat kosong/tidak valid.")

    tidak_wajar = (~df["lat"].between(-90, 90)) | (~df["lon"].between(-180, 180))
    if tidak_wajar.any():
        peringatan.append(f"{int(tidak_wajar.sum())} baris dibuang karena koordinat di luar rentang.")
        df = df[~tidak_wajar].reset_index(drop=True)

    # Deteksi kemungkinan kolom lat/lon tertukar
    if df["lat"].abs().mean() > 60 and df["lon"].abs().mean() < 60:
        df[["lat", "lon"]] = df[["lon", "lat"]].values
        peringatan.append("Kolom latitude & longitude terdeteksi tertukar dan sudah diperbaiki otomatis.")

    if len(df) == 0:
        raise ValueError("Tidak ada data valid setelah dibersihkan.")
    return df, peta, peringatan


def parse_koordinat(teks: str) -> tuple[float, float]:
    """Mengubah teks salinan Google Maps, mis. "-1.634472, 103.492810", menjadi (lat, lon)."""
    bagian = [b for b in str(teks).replace(";", ",").replace(" ", ",").split(",") if b.strip()]
    if len(bagian) != 2:
        raise ValueError("Format koordinat harus 'latitude, longitude', mis. -1.634472, 103.492810")
    try:
        lat, lon = float(bagian[0]), float(bagian[1])
    except ValueError:
        raise ValueError("Koordinat harus berupa angka, mis. -1.634472, 103.492810") from None
    validasi_sekolah((lat, lon))
    return lat, lon


def validasi_sekolah(sekolah: tuple[float, float], df: pd.DataFrame | None = None) -> list[str]:
    """Memeriksa koordinat sekolah. Mengembalikan daftar peringatan (bisa kosong)."""
    lat, lon = sekolah
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Koordinat sekolah di luar rentang (lat -90..90, lon -180..180).")
    peringatan = []
    if df is not None and len(df):
        xy = ke_km_lokal(df["lat"].values, df["lon"].values, lat, lon)
        jarak = np.median(np.hypot(xy[:, 0], xy[:, 1]))
        if jarak > 50:
            peringatan.append(
                f"Sekolah berjarak sekitar {jarak:.0f} km dari alamat siswa. "
                "Periksa kembali koordinat sekolah (mungkin latitude/longitude tertukar)."
            )
    return peringatan


# ---------------------------------------------------------------------------
# 2. Konversi koordinat
# ---------------------------------------------------------------------------
def ke_km_lokal(lat, lon, lat0, lon0):
    """
    Proyeksi equirectangular: ubah (lat, lon) menjadi koordinat datar (km)
    terhadap titik acuan (lat0, lon0) = sekolah, sehingga jarak Euclidean
    sebanding dengan jarak sebenarnya di permukaan bumi (skala kota).
    """
    timur = np.radians(np.asarray(lon) - lon0) * R_BUMI_KM * math.cos(math.radians(lat0))
    utara = np.radians(np.asarray(lat) - lat0) * R_BUMI_KM
    return np.column_stack([timur, utara])


# ---------------------------------------------------------------------------
# 3. Titik jemput unik — siswa dengan koordinat sama digabung
# ---------------------------------------------------------------------------
@dataclass
class Titik:
    xy: np.ndarray                               # (m, 2) koordinat km
    bobot: np.ndarray                            # (m,) jumlah siswa per titik
    anggota: list = field(default_factory=list)  # index siswa per titik


def bentuk_titik(xy_siswa: np.ndarray, kapasitas: int) -> Titik:
    """Gabungkan siswa yang koordinatnya identik (mis. kakak-adik) menjadi satu titik
    agar selalu satu kendaraan. Titik berisi > kapasitas siswa dipecah per kapasitas."""
    grup: dict = {}
    for i, p in enumerate(xy_siswa):
        grup.setdefault(tuple(np.round(p, 6)), []).append(i)
    xy, bobot, anggota = [], [], []
    for idx in grup.values():
        for s in range(0, len(idx), kapasitas):
            potong = idx[s : s + kapasitas]
            xy.append(xy_siswa[potong[0]])
            bobot.append(len(potong))
            anggota.append(potong)
    return Titik(np.array(xy), np.array(bobot), anggota)


def _pusat(t: Titik, label: np.ndarray, k: int) -> np.ndarray:
    pusat = np.zeros((k, 2))
    for c in range(k):
        m = label == c
        if m.any():
            pusat[c] = np.average(t.xy[m], axis=0, weights=t.bobot[m])
    return pusat


def skor_sse(t: Titik, label: np.ndarray, k: int) -> float:
    """Sum of Squared Errors: jumlah jarak kuadrat siswa ke pusat clusternya.
    Semakin kecil, semakin rapat (kompak) cluster."""
    pusat = _pusat(t, label, k)
    return float((((t.xy - pusat[label]) ** 2).sum(1) * t.bobot).sum())


def _isi(t: Titik, label: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(label, weights=t.bobot, minlength=k).astype(int)


# ---------------------------------------------------------------------------
# 4. Penugasan awal
# ---------------------------------------------------------------------------
def penugasan_kmeans(t: Titik, k: int, kapasitas: int, seed: int, maks_iter: int = 50) -> np.ndarray:
    """
    Capacitated K-Means:
      a) Pusat awal dari K-Means biasa (k-means++), dibobot jumlah siswa.
      b) Siswa ditugaskan ke pusat dengan algoritma Hungarian; setiap pusat
         hanya punya `kapasitas` kursi -> batas kapasitas dijamin.
      c) Pusat diperbarui = rata-rata anggota; ulangi sampai stabil.
    """
    idx_titik = np.repeat(np.arange(len(t.bobot)), t.bobot)   # titik -> siswa
    X = t.xy[idx_titik]
    pusat = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(
        t.xy, sample_weight=t.bobot).cluster_centers_.copy()

    label = None
    for _ in range(maks_iter):
        d2 = ((X[:, None, :] - pusat[None, :, :]) ** 2).sum(-1)   # (n, k)
        biaya = np.repeat(d2, kapasitas, axis=1)                    # (n, k*kapasitas)
        baris, kolom = linear_sum_assignment(biaya)
        baru = np.empty(len(X), dtype=int)
        baru[baris] = kolom // kapasitas
        if label is not None and np.array_equal(baru, label):
            break
        label = baru
        for c in range(k):
            if np.any(label == c):
                pusat[c] = X[label == c].mean(0)

    # kembali ke level titik (mayoritas anggota)
    return np.array([np.bincount(label[idx_titik == h], minlength=k).argmax()
                     for h in range(len(t.bobot))])


def penugasan_sweep(t: Titik, k: int, kapasitas: int) -> np.ndarray:
    """
    Sweep: urutkan titik berdasarkan sudut terhadap sekolah (0,0), lalu isi
    kendaraan satu per satu. Semua titik awal sapuan dicoba; dipilih yang
    paling kompak (SSE terkecil).
    """
    urut = np.argsort(np.arctan2(t.xy[:, 1], t.xy[:, 0]))
    target = min(math.ceil(t.bobot.sum() / k), kapasitas)
    terbaik, sse_terbaik = None, np.inf
    for mulai in range(len(urut)):
        label = np.full(len(t.bobot), -1)
        c, isi = 0, 0
        for h in np.roll(urut, -mulai):
            if isi + t.bobot[h] > target and c < k - 1:
                c, isi = c + 1, 0
            label[h] = c
            isi += t.bobot[h]
        if _isi(t, label, k).max() > kapasitas:
            continue
        sse = skor_sse(t, label, k)
        if sse < sse_terbaik:
            terbaik, sse_terbaik = label.copy(), sse
    return terbaik if terbaik is not None else penugasan_kmeans(t, k, kapasitas, 0)


# ---------------------------------------------------------------------------
# 5. Perbaikan (repair kapasitas + local search)
# ---------------------------------------------------------------------------
def perbaiki_kapasitas(t: Titik, label: np.ndarray, k: int, kapasitas: int) -> np.ndarray:
    """Pindahkan titik dari cluster yang kelebihan muatan ke cluster terdekat yang masih muat."""
    label = label.copy()
    for _ in range(10 * len(label)):
        isi = _isi(t, label, k)
        lebih = np.where(isi > kapasitas)[0]
        if len(lebih) == 0:
            break
        c = lebih[0]
        pusat = _pusat(t, label, k)
        kandidat = [
            (np.sum((t.xy[h] - pusat[c2]) ** 2), h, c2)
            for h in np.where(label == c)[0] for c2 in range(k)
            if c2 != c and isi[c2] + t.bobot[h] <= kapasitas
        ]
        if not kandidat:
            raise ValueError("Kapasitas total tidak mencukupi. Tambah jumlah kendaraan.")
        _, h, c2 = min(kandidat)
        label[h] = c2
    return label


def local_search(t: Titik, label: np.ndarray, k: int, kapasitas: int, maks_putaran: int = 30) -> np.ndarray:
    """
    Memperkecil SSE (membuat cluster makin rapat) tanpa melanggar kapasitas:
      - MOVE: pindahkan satu titik ke cluster lain yang masih punya kursi.
      - SWAP: tukar dua titik antar-cluster.
    Perubahan diterima hanya jika SSE turun.
    """
    label = label.copy()
    sse = skor_sse(t, label, k)
    m = len(label)
    for _ in range(maks_putaran):
        membaik = False
        for h in range(m):                                   # MOVE
            isi = _isi(t, label, k)
            asal = label[h]
            for c2 in range(k):
                if c2 == asal or isi[c2] + t.bobot[h] > kapasitas:
                    continue
                label[h] = c2
                s = skor_sse(t, label, k)
                if s < sse - 1e-9:
                    sse, membaik = s, True
                    break
                label[h] = asal
        for a in range(m):                                   # SWAP
            for b in range(a + 1, m):
                ca, cb = label[a], label[b]
                if ca == cb:
                    continue
                isi = _isi(t, label, k)
                if (isi[ca] - t.bobot[a] + t.bobot[b] > kapasitas
                        or isi[cb] - t.bobot[b] + t.bobot[a] > kapasitas):
                    continue
                label[a], label[b] = cb, ca
                s = skor_sse(t, label, k)
                if s < sse - 1e-9:
                    sse, membaik = s, True
                else:
                    label[a], label[b] = ca, cb
        if not membaik:
            break
    return label


# ---------------------------------------------------------------------------
# 6. Fungsi utama
# ---------------------------------------------------------------------------
@dataclass
class HasilClustering:
    detail: pd.DataFrame        # satu baris per siswa
    ringkasan: pd.DataFrame     # satu baris per kendaraan
    metode: str
    sse: float
    sekolah: tuple[float, float]
    kapasitas: int
    log: list[str]


def jalankan_clustering(
    df: pd.DataFrame,
    sekolah: tuple[float, float] = SEKOLAH_DEFAULT,
    kapasitas: int = 10,
    jumlah_kendaraan: int | None = None,
    metode: str = "auto",
    n_percobaan: int = 10,
    seed: int = 42,
) -> HasilClustering:
    """
    df        : DataFrame hasil siapkan_data()
    sekolah   : (lat, lon) sekolah — acuan arah (metode Sweep) & penomoran kendaraan
    kapasitas : maksimal siswa per kendaraan
    jumlah_kendaraan : None = otomatis ceil(n / kapasitas)
    metode    : "auto" | "kmeans" | "sweep"
    """
    log = []
    n = len(df)
    k_min = math.ceil(n / kapasitas)
    k = int(jumlah_kendaraan) if jumlah_kendaraan else k_min
    if k < k_min:
        raise ValueError(f"Jumlah kendaraan minimal untuk {n} siswa adalah {k_min}.")
    k = min(k, n)
    log.append(f"Jumlah siswa: {n} | kapasitas: {kapasitas} | jumlah kendaraan: {k}")

    xy = ke_km_lokal(df["lat"].values, df["lon"].values, *sekolah)
    t = bentuk_titik(xy, kapasitas)
    log.append(f"Jumlah titik jemput unik: {len(t.bobot)}")

    kandidat = []
    if metode in ("auto", "kmeans"):
        for s in range(n_percobaan):
            lab = penugasan_kmeans(t, k, kapasitas, seed + s)
            lab = local_search(t, perbaiki_kapasitas(t, lab, k, kapasitas), k, kapasitas)
            kandidat.append((skor_sse(t, lab, k), "Capacitated K-Means", lab))
    if metode in ("auto", "sweep"):
        lab = penugasan_sweep(t, k, kapasitas)
        lab = local_search(t, perbaiki_kapasitas(t, lab, k, kapasitas), k, kapasitas)
        kandidat.append((skor_sse(t, lab, k), "Sweep", lab))
    if not kandidat:
        raise ValueError("Metode tidak dikenal. Gunakan 'auto', 'kmeans', atau 'sweep'.")

    for nama in sorted({c[1] for c in kandidat}):
        log.append(f"{nama}: SSE terbaik = {min(c[0] for c in kandidat if c[1] == nama):.2f}")
    sse, nama_metode, label = min(kandidat, key=lambda c: c[0])
    log.append(f"Dipilih: {nama_metode} (SSE {sse:.2f}; semakin kecil semakin kompak)")

    # Nomor kendaraan diurutkan searah jarum jam dari arah utara sekolah
    pusat = _pusat(t, label, k)
    sudut = (np.degrees(np.arctan2(pusat[:, 0], pusat[:, 1])) + 360) % 360
    nomor = {c: i + 1 for i, c in enumerate(np.argsort(sudut))}

    detail = df.copy()
    detail.insert(0, "kendaraan", 0)
    for h, c in enumerate(label):
        detail.loc[t.anggota[h], "kendaraan"] = nomor[c]
    detail = detail.sort_values(["kendaraan", "alamat", "id_siswa"]).reset_index(drop=True)

    ringkasan = (
        detail.groupby("kendaraan")
        .agg(jumlah_siswa=("id_siswa", "size"),
             jumlah_titik_jemput=("alamat", lambda s: detail.loc[s.index, ["lat", "lon"]].drop_duplicates().shape[0]),
             pusat_lat=("lat", "mean"), pusat_lon=("lon", "mean"),
             daftar_alamat=("alamat", lambda s: ", ".join(dict.fromkeys(s))))
        .reset_index()
    )
    ringkasan[["pusat_lat", "pusat_lon"]] = ringkasan[["pusat_lat", "pusat_lon"]].round(6)
    return HasilClustering(detail, ringkasan, nama_metode, round(sse, 2), sekolah, kapasitas, log)


# ---------------------------------------------------------------------------
# 7. Ekspor
# ---------------------------------------------------------------------------
def ke_excel_bytes(hasil: HasilClustering) -> bytes:
    """Menulis hasil ke file Excel (3 sheet) dan mengembalikan isinya sebagai bytes."""
    buf = io.BytesIO()
    info = pd.DataFrame({
        "parameter": ["Metode terpilih", "Kapasitas per kendaraan", "Jumlah kendaraan",
                      "Jumlah siswa", "Latitude sekolah", "Longitude sekolah"],
        "nilai": [hasil.metode, hasil.kapasitas, len(hasil.ringkasan), len(hasil.detail),
                  hasil.sekolah[0], hasil.sekolah[1]],
    })
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        hasil.detail.to_excel(w, sheet_name="Hasil_Cluster", index=False)
        hasil.ringkasan.to_excel(w, sheet_name="Ringkasan_Kendaraan", index=False)
        info.to_excel(w, sheet_name="Info", index=False)
        for ws in w.book.worksheets:
            for col in ws.columns:
                lebar = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, lebar + 2), 80)
            ws.freeze_panes = "A2"
    return buf.getvalue()


def buat_template_bytes() -> bytes:
    """Template Excel dengan contoh baris."""
    contoh = pd.DataFrame({
        "alamat": ["Contoh: Perum Citraland", "Contoh: Kota Baru Indah"],
        "x": [-1.649954, -1.639010],
        "y": [103.579371, 103.544126],
    })
    buf = io.BytesIO()
    contoh.to_excel(buf, index=False)
    return buf.getvalue()
