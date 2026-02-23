import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Dashboard Analisis Soal Siswa",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Dashboard Analisis Jawaban Siswa")
st.markdown("Analisis berbasis data hasil jawaban 100 siswa terhadap 20 soal")

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data_simulasi_50_siswa_20_soal_xlsx.xlsx")
    return df

df = load_data()

# Ambil semua 20 soal dan pastikan numerik
soal_cols = [f"Soal_{i}" for i in range(1, 21)]
data_soal = df[soal_cols].apply(pd.to_numeric, errors="coerce")

# ==========================================================
# KPI UTAMA
# ==========================================================
mean_scores = data_soal.mean()
skor_rata = mean_scores.mean()
# Skala max = 4 (berdasarkan data)
ikm = (skor_rata / 4) * 100

def kategori_ikm(x):
    if x >= 81: return "Sangat Baik"
    elif x >= 66: return "Baik"
    elif x >= 51: return "Cukup"
    else: return "Kurang"

col1, col2, col3, col4 = st.columns(4)
col1.metric("📈 Indeks Kepuasan Rata-rata", f"{ikm:.2f}%")
col2.metric("🏷️ Kategori", kategori_ikm(ikm))
col3.metric("👥 Jumlah Responden (Siswa)", len(df))
col4.metric("📝 Jumlah Soal", 20)

st.divider()

# ==========================================================
# 1️⃣ DISTRIBUSI SKOR PER SOAL
# ==========================================================
st.header("1️⃣ Rata-rata Skor per Soal")

fig1, ax1 = plt.subplots(figsize=(14, 5))
colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, 20))
bars = ax1.bar(soal_cols, mean_scores.values, color=colors_bar, edgecolor="white", linewidth=0.5)
ax1.axhline(skor_rata, linestyle="--", color="#e74c3c", linewidth=1.5, label=f"Rata-rata keseluruhan: {skor_rata:.2f}")
ax1.set_ylabel("Rata-rata Skor (Skala 1–4)")
ax1.set_xlabel("Soal")
ax1.set_title("Rata-rata Skor Jawaban per Soal", fontsize=14, fontweight="bold")
ax1.set_ylim(0, 4.5)
ax1.set_xticklabels(soal_cols, rotation=45, ha="right", fontsize=8)
ax1.legend()
ax1.grid(axis="y", linestyle="--", alpha=0.4)

for bar, val in zip(bars, mean_scores.values):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.06, f"{val:.2f}",
             ha="center", fontsize=7, fontweight="bold")

st.pyplot(fig1)

soal_terendah = mean_scores.idxmin()
soal_tertinggi = mean_scores.idxmax()
c1, c2 = st.columns(2)
c1.warning(f"⚠️ Soal dengan skor terendah: **{soal_terendah}** ({mean_scores[soal_terendah]:.2f})")
c2.success(f"✅ Soal dengan skor tertinggi: **{soal_tertinggi}** ({mean_scores[soal_tertinggi]:.2f})")

st.divider()

# ==========================================================
# 2️⃣ DISTRIBUSI FREKUENSI SKOR (HISTOGRAM)
# ==========================================================
st.header("2️⃣ Distribusi Frekuensi Jawaban")

col_left, col_right = st.columns([1, 2])

with col_left:
    soal_pilih = st.selectbox("Pilih soal untuk melihat distribusi:", soal_cols)

with col_right:
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    nilai_counts = data_soal[soal_pilih].value_counts().sort_index()
    warna_hist = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    ax2.bar(nilai_counts.index, nilai_counts.values,
            color=warna_hist[:len(nilai_counts)], edgecolor="white", width=0.6)
    ax2.set_xlabel("Nilai Jawaban")
    ax2.set_ylabel("Jumlah Siswa")
    ax2.set_title(f"Distribusi Jawaban – {soal_pilih}", fontweight="bold")
    ax2.set_xticks(nilai_counts.index)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for x, y in zip(nilai_counts.index, nilai_counts.values):
        ax2.text(x, y + 0.5, str(y), ha="center", fontweight="bold")
    st.pyplot(fig2)

# Ringkasan statistik deskriptif
st.subheader("📊 Statistik Deskriptif Semua Soal")
stats_df = data_soal.describe().T.round(2)
stats_df.index.name = "Soal"
st.dataframe(stats_df, use_container_width=True)

st.divider()

# ==========================================================
# 3️⃣ ANALISIS GAP (SKOR vs SKOR MAKSIMUM)
# ==========================================================
st.header("3️⃣ Analisis GAP (Skor Aktual vs Skor Maksimum)")

gap_scores = 4 - mean_scores  # Skala max = 4
prioritas_gap = gap_scores.idxmax()

fig3, ax3 = plt.subplots(figsize=(14, 5))
colors_gap = ["#e74c3c" if v == gap_scores.max() else "#3498db" for v in gap_scores.values]
ax3.bar(soal_cols, gap_scores.values, color=colors_gap, edgecolor="white")
ax3.set_ylabel("Nilai GAP (Jarak ke Skor Maksimum)")
ax3.set_xlabel("Soal")
ax3.set_title("GAP Skor per Soal (Skor Maks = 4)", fontsize=14, fontweight="bold")
ax3.set_xticklabels(soal_cols, rotation=45, ha="right", fontsize=8)
ax3.grid(axis="y", linestyle="--", alpha=0.4)

for i, v in enumerate(gap_scores.values):
    ax3.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7, fontweight="bold")

patch_merah = mpatches.Patch(color="#e74c3c", label="Prioritas perbaikan utama")
patch_biru = mpatches.Patch(color="#3498db", label="Soal lainnya")
ax3.legend(handles=[patch_merah, patch_biru])

st.pyplot(fig3)
st.error(f"📌 Soal dengan GAP terbesar (prioritas perbaikan): **{prioritas_gap}** (GAP = {gap_scores[prioritas_gap]:.2f})")

st.divider()

# ==========================================================
# 4️⃣ HEATMAP KORELASI ANTAR SOAL
# ==========================================================
st.header("4️⃣ Korelasi Antar Soal")

corr = data_soal.corr()

fig4, ax4 = plt.subplots(figsize=(14, 11))
im = ax4.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax4, fraction=0.03)

ax4.set_xticks(range(20))
ax4.set_yticks(range(20))
ax4.set_xticklabels([f"S{i}" for i in range(1, 21)], rotation=45, ha="right", fontsize=8)
ax4.set_yticklabels([f"S{i}" for i in range(1, 21)], fontsize=8)

for i in range(20):
    for j in range(20):
        val = corr.iloc[i, j]
        color_text = "white" if abs(val) > 0.6 else "black"
        ax4.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color=color_text)

ax4.set_title("Heatmap Korelasi Pearson antar 20 Soal", fontsize=14, fontweight="bold")
st.pyplot(fig4)

# Ranking korelasi dengan soal terakhir (Soal_20) sebagai referensi
st.subheader("📊 Ranking Korelasi dengan Soal_20")
corr_s20 = corr["Soal_20"].drop("Soal_20").sort_values(ascending=False)
st.dataframe(corr_s20.to_frame("Koefisien Korelasi").round(3), use_container_width=True)

st.divider()

# ==========================================================
# 5️⃣ ANALISIS REGRESI LINEAR BERGANDA
# ==========================================================
st.header("5️⃣ Analisis Regresi Linear Berganda")
st.markdown("Soal_20 sebagai variabel dependen (Y), Soal_1 s.d. Soal_19 sebagai prediktor (X)")

X_reg = sm.add_constant(data_soal.iloc[:, 0:19])
y_reg = data_soal["Soal_20"]

model = sm.OLS(y_reg, X_reg, missing="drop").fit()
coef = model.params[1:]
r2 = model.rsquared

fig5, ax5 = plt.subplots(figsize=(14, 5))
colors_reg = ["#e74c3c" if v > 0 else "#3498db" for v in coef.values]
ax5.bar(coef.index, coef.values, color=colors_reg, edgecolor="white")
ax5.axhline(0, linestyle="--", color="black", linewidth=1)
ax5.set_xticklabels(coef.index, rotation=45, ha="right", fontsize=8)
ax5.set_ylabel("Koefisien Regresi")
ax5.set_title("Koefisien Regresi – Prediktor terhadap Soal_20", fontsize=13, fontweight="bold")
ax5.grid(axis="y", linestyle="--", alpha=0.4)

patch_pos = mpatches.Patch(color="#e74c3c", label="Pengaruh Positif")
patch_neg = mpatches.Patch(color="#3498db", label="Pengaruh Negatif")
ax5.legend(handles=[patch_pos, patch_neg])

st.pyplot(fig5)

col_r2, col_dom = st.columns(2)
col_r2.info(f"📈 Nilai R²: **{r2:.4f}** — model menjelaskan {r2*100:.1f}% variansi Soal_20")
col_dom.success(f"🔑 Faktor dominan: **{coef.abs().idxmax()}** (koef = {coef[coef.abs().idxmax()]:.4f})")

# Tampilkan tabel koefisien
st.subheader("📋 Tabel Koefisien Regresi Lengkap")
coef_df = pd.DataFrame({
    "Koefisien": model.params[1:].round(4),
    "Std. Error": model.bse[1:].round(4),
    "t-value": model.tvalues[1:].round(3),
    "p-value": model.pvalues[1:].round(4)
})
st.dataframe(coef_df, use_container_width=True)

st.divider()

# ==========================================================
# 6️⃣ SEGMENTASI SISWA (K-MEANS CLUSTERING)
# ==========================================================
st.header("6️⃣ Segmentasi Siswa berdasarkan Pola Jawaban")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_soal.fillna(data_soal.mean()))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

data_cluster = data_soal.copy()
data_cluster["Cluster"] = cluster_labels

cluster_mean = data_cluster.groupby("Cluster")[soal_cols].mean()
cluster_mean = cluster_mean.sort_values(by="Soal_20", ascending=False)
cluster_mean["Segmen"] = ["Kemampuan Tinggi", "Kemampuan Sedang", "Kemampuan Rendah"]

# Statistik cluster
st.subheader("📊 Statistik Per Segmen")
cluster_stats = data_cluster.groupby("Cluster").agg(
    Jumlah_Siswa=("Soal_1", "count"),
)
cluster_stats["Rata_Rata_Skor"] = data_cluster.groupby("Cluster")[soal_cols].mean().mean(axis=1).round(2)
cluster_stats.index = ["Segmen " + str(i) for i in cluster_stats.index]
st.dataframe(cluster_stats, use_container_width=True)

# Radar Chart – pilih 8 soal representatif agar tidak terlalu padat
radar_soal = ["Soal_1","Soal_3","Soal_5","Soal_7","Soal_10","Soal_13","Soal_16","Soal_20"]
labels_radar = radar_soal
angles = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
angles += angles[:1]

fig6 = plt.figure(figsize=(7, 7))
ax6 = plt.subplot(111, polar=True)

colors_radar = ["#2ecc71", "#f39c12", "#e74c3c"]
segmen_names = ["Kemampuan Tinggi", "Kemampuan Sedang", "Kemampuan Rendah"]

for idx, (cluster_idx, row) in enumerate(cluster_mean.iterrows()):
    values = [row[s] for s in radar_soal] + [row[radar_soal[0]]]
    ax6.plot(angles, values, color=colors_radar[idx], linewidth=2, label=segmen_names[idx])
    ax6.fill(angles, values, color=colors_radar[idx], alpha=0.15)

ax6.set_thetagrids(np.degrees(angles[:-1]), labels_radar, fontsize=9)
ax6.set_ylim(0, 4)
ax6.set_title("Radar Chart – Segmentasi Kemampuan Siswa\n(8 Soal Representatif)", fontsize=12, fontweight="bold", pad=20)
ax6.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

st.pyplot(fig6)

# Bar chart jumlah siswa per segmen
fig7, ax7 = plt.subplots(figsize=(6, 3))
jumlah_per_cluster = data_cluster["Cluster"].value_counts().sort_index()
segmen_label = ["Kemampuan Tinggi", "Kemampuan Sedang", "Kemampuan Rendah"]
# Urutkan sesuai cluster_mean index
ordered_labels = [segmen_names[i] for i in range(3)]
ordered_counts = [jumlah_per_cluster.get(cluster_mean.index[i], 0) for i in range(3)]

ax7.bar(ordered_labels, ordered_counts, color=colors_radar, edgecolor="white")
ax7.set_ylabel("Jumlah Siswa")
ax7.set_title("Distribusi Siswa per Segmen", fontweight="bold")
ax7.grid(axis="y", linestyle="--", alpha=0.4)
for i, v in enumerate(ordered_counts):
    ax7.text(i, v + 0.3, str(v), ha="center", fontweight="bold")

st.pyplot(fig7)
st.success("📌 Segmentasi berhasil! Guru dapat memberikan intervensi sesuai kategori kemampuan siswa.")

st.divider()

# ==========================================================
# 7️⃣ TABEL DATA LENGKAP
# ==========================================================
st.header("7️⃣ Tabel Data Lengkap")

df_tampil = df.copy()
df_tampil["Total Skor"] = data_soal.sum(axis=1)
df_tampil["Rata-rata"] = data_soal.mean(axis=1).round(2)
df_tampil["Segmen"] = [segmen_names[cluster_mean.index.tolist().index(c)] if c in cluster_mean.index else "-"
                        for c in cluster_labels]

st.dataframe(df_tampil, use_container_width=True)

# Download data
csv = df_tampil.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Data sebagai CSV",
    data=csv,
    file_name="hasil_analisis_siswa.csv",
    mime="text/csv"
)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown("🎓 **Dashboard Analisis Soal Siswa** | Dibuat dengan Streamlit | Mata Kuliah Fisika Komputasi")
