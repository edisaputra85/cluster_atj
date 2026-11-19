"""
clustering_antar_jemput.py
Versi skrip (tanpa antarmuka) — cocok dijalankan dari VSCode / terminal.

Contoh:
    python clustering_antar_jemput.py DataSetAlamat_clean.xlsx --sekolah -1.634472 103.492810
    python clustering_antar_jemput.py DataSetAlamat_clean.xlsx --kapasitas 10 --metode auto --kendaraan 10

Jika --sekolah tidak diisi, dipakai SEKOLAH_DEFAULT pada clustering_core.py.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import clustering_core as cc


def gambar_hasil(hasil, file_png):
    lat_s, lon_s = hasil.sekolah
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20")
    for i, (kend, g) in enumerate(hasil.detail.groupby("kendaraan")):
        ax.scatter(g["lon"], g["lat"], s=80, color=cmap(i % 20), edgecolor="k", lw=0.5,
                   label=f"Mobil {kend} ({len(g)} siswa)", zorder=3)
    ax.scatter([lon_s], [lat_s], s=300, marker="*", color="red", edgecolor="k", zorder=4, label="Sekolah")
    ax.set_xlabel("Longitude (y)")
    ax.set_ylabel("Latitude (x)")
    ax.set_title(f"Hasil Clustering Antar Jemput — {hasil.metode}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(file_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Clustering alamat antar jemput sekolah")
    p.add_argument("file", help="File Excel/CSV data alamat (kolom: alamat, x, y)")
    p.add_argument("--sekolah", type=float, nargs=2, metavar=("LAT", "LON"), default=cc.SEKOLAH_DEFAULT,
                   help="Koordinat sekolah, mis. --sekolah -1.634472 103.492810")
    p.add_argument("--kapasitas", type=int, default=10)
    p.add_argument("--kendaraan", type=int, default=None, help="Default: otomatis (minimum)")
    p.add_argument("--metode", choices=["auto", "kmeans", "sweep"], default="auto")
    p.add_argument("--output", default="hasil_clustering_antar_jemput.xlsx")
    a = p.parse_args()

    df_mentah = pd.read_csv(a.file) if a.file.lower().endswith(".csv") else pd.read_excel(a.file)
    df, _, peringatan = cc.siapkan_data(df_mentah)
    sekolah = tuple(a.sekolah)
    for w in peringatan + cc.validasi_sekolah(sekolah, df):
        print("PERINGATAN:", w)
    print(f"Lokasi sekolah: {sekolah[0]:.6f}, {sekolah[1]:.6f}")

    hasil = cc.jalankan_clustering(df, sekolah, kapasitas=a.kapasitas,
                                   jumlah_kendaraan=a.kendaraan, metode=a.metode)
    print("\n".join(hasil.log))
    print()
    print(hasil.ringkasan[["kendaraan", "jumlah_siswa", "daftar_alamat"]].to_string(index=False))

    with open(a.output, "wb") as f:
        f.write(cc.ke_excel_bytes(hasil))
    png = a.output.rsplit(".", 1)[0] + ".png"
    gambar_hasil(hasil, png)
    print(f"\nHasil disimpan: {a.output} dan {png}")


if __name__ == "__main__":
    main()
