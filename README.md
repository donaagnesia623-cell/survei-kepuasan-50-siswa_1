# 📚 Dashboard Analisis Jawaban Siswa

Dashboard interaktif berbasis **Streamlit** untuk menganalisis hasil jawaban 100 siswa terhadap 20 soal.

> Tugas Mata Kuliah Fisika Komputasi

## 🔗 Link Demo
> *(Isi dengan link Streamlit Cloud kamu setelah deploy)*

---

## 📊 Fitur Dashboard

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | **KPI Utama** | Indeks kepuasan rata-rata, kategori, jumlah responden |
| 2 | **Distribusi Skor per Soal** | Bar chart rata-rata skor tiap soal + soal tertinggi/terendah |
| 3 | **Distribusi Frekuensi** | Histogram jawaban per soal (bisa dipilih) + statistik deskriptif |
| 4 | **Analisis GAP** | Jarak skor aktual ke skor maksimum per soal |
| 5 | **Heatmap Korelasi** | Korelasi Pearson antar 20 soal |
| 6 | **Regresi Linear Berganda** | Soal_20 sebagai Y, Soal_1–19 sebagai X |
| 7 | **Segmentasi K-Means** | Pengelompokan siswa: Kemampuan Tinggi/Sedang/Rendah |
| 8 | **Tabel Data Lengkap** | Data lengkap + download CSV |

---

## 🗂️ Struktur File

```
├── dashboard_siswa.py                        # File utama Streamlit
├── data_simulasi_50_siswa_20_soal_xlsx.xlsx  # Dataset
├── requirements.txt                          # Dependensi Python
└── README.md                                 # Dokumentasi ini
```

---

## 🚀 Cara Menjalankan Secara Lokal

### 1. Clone Repository
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Dashboard
```bash
streamlit run dashboard_siswa.py
```

Dashboard akan terbuka otomatis di browser: `http://localhost:8501`

---

## ☁️ Deploy ke Streamlit Cloud

1. Push semua file ke GitHub (pastikan file Excel ikut di-upload)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan akun GitHub
4. Klik **"New app"** → pilih repo dan file `dashboard_siswa.py`
5. Klik **Deploy** → tunggu beberapa menit
6. Salin link yang diberikan dan tempel di README ini

---

## 📦 Library yang Digunakan

- `streamlit` — framework dashboard interaktif
- `pandas` — manipulasi data
- `numpy` — komputasi numerik
- `matplotlib` — visualisasi data
- `scikit-learn` — clustering (K-Means) & preprocessing
- `statsmodels` — regresi linear berganda (OLS)
- `openpyxl` — membaca file Excel

---

## 📝 Tentang Data

- **Jumlah responden:** 100 siswa (R1–R100)
- **Jumlah soal:** 20 soal (Soal_1 – Soal_20)
- **Skala jawaban:** 1–4
- **Format file:** `.xlsx`
