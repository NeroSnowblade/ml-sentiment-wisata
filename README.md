# Aplikasi Analisis Sentimen Ulasan Wisata (Flask + Random Forest)

Aplikasi web sederhana untuk menganalisis sentimen (positif, negatif, netral) dari ulasan wisata berbahasa Indonesia. Model menggunakan scikit-learn RandomForest dengan pipeline TF-IDF dan normalisasi Bahasa Indonesia (slang/abbr/stopwords/stemming Sastrawi). Dashboard menampilkan total ulasan, grafik bar, pie, serta word cloud per sentimen.

## Fitur
- Input ulasan dan analisis sentimen instan.
- Dashboard:
  - Total ulasan
  - Bar chart jumlah per sentimen
  - Pie chart persentase sentimen
  - Word Cloud untuk positif / negatif / netral
- Preprocessing Bahasa Indonesia:
  - Normalisasi slang & singkatan (kamus lokal)
  - Stopwords (lokal) & stemming (Sastrawi)
  - Penanganan URL, emoji, tanda baca
  - Deteksi bahasa (langdetect) sebagai helper untuk komentar campuran bahasa
- UI dengan Bootstrap 5, tema warna:
  - primary: #ffc8dd
  - secondary: #bce0ff / #a2d2ff
  - tertiary: #c5aced / #f7abeb

## Struktur Folder
```
app.py                      # Aplikasi Flask (routes, DB, model)
preprocess.py               # Normalisasi & tokenizer + wordcloud helper
models/sentiment_rf.pkl     # Model tersimpan (otomatis dibuat saat pertama run)
static/css/theme.css        # Tema Bootstrap kustom
static/wordcloud/*.png      # Gambar word cloud hasil generate
templates/*.html            # Template Dashboard & Input
data/
  ├─ ulasan_labeled.csv     # Dataset contoh untuk training & seeding
  ├─ stopwords_id.txt       # Stopwords Indonesia lokal
  ├─ slang_indonesia.json   # Kamus slang
  └─ abbreviations.json     # Kamus singkatan
requirements.txt            # Dependensi Python
```

## Cara Menjalankan (Windows PowerShell)

Opsional: aktifkan virtual environment lebih dulu.

```powershell
# 1) (Opsional) Buat dan aktifkan venv
python -m venv .venv ; .\.venv\Scripts\Activate.ps1

# 2) Install dependensi
pip install -r requirements.txt

# 3) Jalankan aplikasi
python app.py
```

Aplikasi akan tersedia di: http://127.0.0.1:5000/

Catatan:
- Pada run pertama, model akan dilatih dari `data/ulasan_labeled.csv` dan database `reviews.db` akan diinisialisasi serta di-seed dari dataset tersebut agar dashboard langsung berisi data.
- Word cloud akan dibuat ulang saat membuka dashboard.

## Catatan Teknis
- Model: `TfidfVectorizer` (tokenizer khusus Bahasa Indonesia) + `RandomForestClassifier(n_estimators=300)`.
- Tokenizer: `preprocess.tokenize_for_ml` yang melakukan normalisasi slang/singkatan/stopwords/stemming.
- Database: SQLite (file `reviews.db`) dengan tabel `reviews`.
- Grafik: Chart.js (CDN) untuk bar dan pie chart; WordCloud dihasilkan server-side dengan library `wordcloud`.

## Kustomisasi
- Tambahkan/ubah kamus slang/singkatan di `data/slang_indonesia.json` dan `data/abbreviations.json`.
- Tambahkan stopwords di `data/stopwords_id.txt`.
- Ubah tema Bootstrap di `static/css/theme.css`.

## Troubleshooting
- Jika instalasi `wordcloud` gagal di Windows, pastikan Python dan pip up-to-date. Coba `pip install --upgrade pip wheel setuptools` lalu install lagi.
- Jika `langdetect` error pada teks sangat pendek, pipeline tetap berjalan; deteksi bahasa bersifat best-effort.
- Jika font unicode bermasalah untuk WordCloud, install font yang mendukung bahasa Indonesia dan set `font_path` manual di `preprocess.generate_wordcloud`.
