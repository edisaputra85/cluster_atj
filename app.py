"""
app.py — Aplikasi Streamlit Clustering Antar Jemput Sekolah
Jalankan:  streamlit run app.py
"""

import io

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import clustering_core as cc

st.set_page_config(page_title="Clustering Antar Jemput Sekolah", page_icon="🚐", layout="wide")

st.title("🚐 Clustering Alamat Antar Jemput Sekolah")
st.caption(
    "Unggah data alamat siswa (Excel), atur kapasitas kendaraan, lalu sistem akan "
    "mengelompokkan alamat yang berdekatan ke dalam satu kendaraan."
)

# ---------------------------------------------------------------------------
# Sidebar: pengaturan
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.download_button(
        "📄 Unduh template Excel",
        data=cc.buat_template_bytes(),
        file_name="template_dataset_alamat.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    st.divider()
    kapasitas = st.number_input("Kapasitas per kendaraan (siswa)", 1, 50, 10, 1)
    mode_k = st.radio("Jumlah kendaraan", ["Otomatis (minimum)", "Tentukan sendiri"])
    k_manual = None
    if mode_k == "Tentukan sendiri":
        k_manual = st.number_input("Jumlah kendaraan", 1, 200, 9, 1)

    metode_label = st.selectbox(
        "Metode clustering",
        ["Otomatis (pilih paling kompak)", "Capacitated K-Means", "Sweep (sudut dari sekolah)"],
    )
    metode = {"Otomatis (pilih paling kompak)": "auto",
              "Capacitated K-Means": "kmeans",
              "Sweep (sudut dari sekolah)": "sweep"}[metode_label]
    n_percobaan = st.slider("Jumlah percobaan K-Means (acak)", 1, 30, 10,
                            help="Semakin banyak, hasil cenderung lebih baik tetapi lebih lama.")
    seed = st.number_input("Random seed", 0, 9999, 42, 1)

# ---------------------------------------------------------------------------
# 1. Upload data
# ---------------------------------------------------------------------------
st.subheader("1️⃣ Unggah Data")
berkas = st.file_uploader(
    "File Excel (.xlsx) dengan kolom: alamat, x (latitude), y (longitude)",
    type=["xlsx", "xls", "csv"],
)

if berkas is None:
    st.info("Silakan unggah file data alamat. Gunakan tombol **Unduh template Excel** di sidebar bila perlu contoh format.")
    st.stop()

try:
    if berkas.name.lower().endswith(".csv"):
        df_mentah = pd.read_csv(berkas)
    else:
        df_mentah = pd.read_excel(berkas)
    df, peta, peringatan = cc.siapkan_data(df_mentah)
except Exception as e:
    st.error(f"Gagal membaca data: {e}")
    st.stop()

for p in peringatan:
    st.warning(p)

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah siswa", len(df))
c2.metric("Titik jemput unik", df[["lat", "lon"]].drop_duplicates().shape[0])
c3.metric("Kendaraan minimum", -(-len(df) // kapasitas))

with st.expander("Lihat data yang diunggah", expanded=False):
    st.caption("Pemetaan kolom: " + ", ".join(f"{k} ← '{v}'" for k, v in peta.items()))
    st.dataframe(df, width="stretch", height=300)

# ---------------------------------------------------------------------------
# 2. Lokasi sekolah
# ---------------------------------------------------------------------------
st.subheader("2️⃣ Koordinat Sekolah")
st.caption(
    "Cara mendapatkan koordinat: buka Google Maps → klik kanan pada lokasi sekolah → "
    "klik angka koordinat yang muncul (otomatis tersalin) → tempel di kolom di bawah."
)
if "lat_s" not in st.session_state:
    st.session_state["lat_s"], st.session_state["lon_s"] = cc.SEKOLAH_DEFAULT


def _terapkan_tempel():
    try:
        lat, lon = cc.parse_koordinat(st.session_state["teks_koordinat"])
        st.session_state["lat_s"], st.session_state["lon_s"] = lat, lon
        st.session_state["pesan_koordinat"] = None
    except Exception as e:
        st.session_state["pesan_koordinat"] = str(e)


st.text_input(
    "Tempel koordinat dari Google Maps (format: latitude, longitude)",
    key="teks_koordinat", placeholder="-1.634472, 103.492810", on_change=_terapkan_tempel,
)
if st.session_state.get("pesan_koordinat"):
    st.error(st.session_state["pesan_koordinat"])

s1, s2 = st.columns(2)
lat_s = s1.number_input("Latitude sekolah (x)", min_value=-90.0, max_value=90.0,
                        format="%.6f", step=0.0001, key="lat_s")
lon_s = s2.number_input("Longitude sekolah (y)", min_value=-180.0, max_value=180.0,
                        format="%.6f", step=0.0001, key="lon_s")

for p in cc.validasi_sekolah((lat_s, lon_s), df):
    st.warning(p)

with st.expander("🗺️ Pratinjau lokasi sekolah & alamat siswa", expanded=True):
    pratinjau = go.Figure()
    pratinjau.add_trace(go.Scattermap(lat=df["lat"], lon=df["lon"], mode="markers",
                                      marker=dict(size=9, color="#1f77b4"), text=df["alamat"],
                                      name="Alamat siswa", hovertemplate="%{text}<extra></extra>"))
    pratinjau.add_trace(go.Scattermap(lat=[lat_s], lon=[lon_s], mode="markers",
                                      marker=dict(size=20, color="red"), name="🏫 Sekolah",
                                      hovertemplate="Sekolah<extra></extra>"))
    pratinjau.update_layout(map=dict(style="open-street-map", zoom=11,
                                     center=dict(lat=float(df["lat"].mean()), lon=float(df["lon"].mean()))),
                            height=400, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(pratinjau, width="stretch")
    st.caption("Pastikan titik merah tepat berada di lokasi sekolah sebelum menjalankan clustering.")

# ---------------------------------------------------------------------------
# 3. Proses
# ---------------------------------------------------------------------------
st.subheader("3️⃣ Proses Clustering")
if st.button("🚀 Jalankan Clustering", type="primary", width="stretch"):
    with st.spinner("Mengelompokkan alamat..."):
        try:
            hasil = cc.jalankan_clustering(
                df, (lat_s, lon_s), kapasitas=int(kapasitas),
                jumlah_kendaraan=int(k_manual) if k_manual else None,
                metode=metode, n_percobaan=int(n_percobaan), seed=int(seed),
            )
            st.session_state["hasil"] = hasil
        except Exception as e:
            st.error(f"Proses gagal: {e}")
            st.stop()

hasil = st.session_state.get("hasil")
if hasil is None:
    st.stop()
lat_s, lon_s = hasil.sekolah   # gunakan koordinat sekolah saat clustering dijalankan

# ---------------------------------------------------------------------------
# 4. Hasil
# ---------------------------------------------------------------------------
st.subheader("4️⃣ Hasil")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Metode terpilih", hasil.metode)
m2.metric("Jumlah kendaraan", len(hasil.ringkasan))
m3.metric("Muatan maksimum", f"{hasil.ringkasan['jumlah_siswa'].max()} siswa")
m4.metric("Muatan minimum", f"{hasil.ringkasan['jumlah_siswa'].min()} siswa")

tab_peta, tab_ringkas, tab_detail, tab_log = st.tabs(
    ["🗺️ Peta", "📋 Ringkasan Kendaraan", "👦 Detail Siswa", "🧾 Log Proses"]
)

with tab_peta:
    det = hasil.detail
    warna = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    fig = go.Figure()
    for i, (kend, g) in enumerate(det.groupby("kendaraan")):
        fig.add_trace(go.Scattermap(
            lat=g["lat"], lon=g["lon"], mode="markers+text",
            marker=dict(size=13, color=warna[i % len(warna)]),
            text=[str(kend)] * len(g), textfont=dict(color="white", size=9),
            name=f"Mobil {kend} ({len(g)} siswa)",
            customdata=g[["alamat", "id_siswa"]],
            hovertemplate="<b>%{customdata[0]}</b><br>Siswa #%{customdata[1]}<br>"
                          f"Mobil {kend}<extra></extra>",
        ))
    fig.add_trace(go.Scattermap(
        lat=[lat_s], lon=[lon_s], mode="markers", marker=dict(size=20, color="red"),
        name="🏫 Sekolah", hovertemplate="Sekolah<extra></extra>",
    ))
    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=det["lat"].mean(), lon=det["lon"].mean()), zoom=11.5),
        height=650, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Warna dan angka pada titik = nomor kendaraan. Urutan penjemputan ditentukan oleh pengemudi sesuai kondisi jalan.")

with tab_ringkas:
    st.dataframe(hasil.ringkasan, width="stretch", hide_index=True)
    st.bar_chart(hasil.ringkasan.set_index("kendaraan")["jumlah_siswa"])

with tab_detail:
    pilih = st.multiselect("Filter kendaraan", sorted(hasil.detail["kendaraan"].unique()))
    tampil = hasil.detail if not pilih else hasil.detail[hasil.detail["kendaraan"].isin(pilih)]
    st.dataframe(tampil, width="stretch", hide_index=True, height=450)

with tab_log:
    st.code("\n".join(hasil.log))

# ---------------------------------------------------------------------------
# 5. Unduh
# ---------------------------------------------------------------------------
st.subheader("5️⃣ Unduh Hasil")
d1, d2, d3 = st.columns(3)
d1.download_button(
    "⬇️ Excel (lengkap, 3 sheet)", data=cc.ke_excel_bytes(hasil),
    file_name="hasil_clustering_antar_jemput.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    width="stretch",
)
d2.download_button(
    "⬇️ CSV detail siswa", data=hasil.detail.to_csv(index=False).encode("utf-8-sig"),
    file_name="hasil_clustering_detail.csv", mime="text/csv", width="stretch",
)
d3.download_button(
    "⬇️ CSV ringkasan kendaraan", data=hasil.ringkasan.to_csv(index=False).encode("utf-8-sig"),
    file_name="hasil_clustering_ringkasan.csv", mime="text/csv", width="stretch",
)
