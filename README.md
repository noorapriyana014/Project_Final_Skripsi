# Deteksi Phishing AI 🛡️

## Deskripsi Proyek
Aplikasi berbasis web untuk mendeteksi dan mengklasifikasikan email ke dalam kategori aman (*safe email*) atau ancaman (*phishing*). Proyek ini dibangun untuk mengotomatisasi identifikasi serangan siber menggunakan pemrosesan bahasa alami (NLP) dan Machine Learning.

## Fitur Utama
- **Klasifikasi Cepat:** Memprediksi status teks email dalam hitungan detik.
- **Antarmuka Interaktif:** Dibangun menggunakan Streamlit agar mudah diakses oleh pengguna tanpa perlu menjalankan *script* melalui terminal.

## Teknologi & Metodologi
- **Bahasa Pemrograman:** Python
- **Framework Aplikasi:** Streamlit
- **Machine Learning:** Logistic Regression (via Scikit-learn)
- **Pemrosesan Teks & Data:** Lemmatization, TF-IDF (Term Frequency-Inverse Document Frequency)
- **Penanganan Data Tidak Seimbang:** SMOTE

## Evaluasi & Performa
Sistem ini telah melalui tahap uji coba menggunakan 100 data uji yang terdiri dari 50 *safe email* dan 50 email *phishing*. Hasil dari uji coba tersebut menunjukkan bahwa model berhasil mendeteksi seluruh data dengan benar (**Akurasi 100%**).

## Panduan Instalasi
Untuk menjalankan proyek ini secara lokal, ikuti langkah berikut:
1. Clone repositori ini: `git clone https://github.com/noorapriyana014/phishing-email-detection.git`
2. Install dependensi (pastikan library seperti scikit-learn, streamlit, imbalanced-learn sudah terpasang): `pip install -r requirements.txt`
3. Jalankan aplikasi: `streamlit run app.py`

## Tangkapan Layar (Screenshots)
Berikut adalah tampilan aplikasi saat mendeteksi teks yang terindikasi *phishing*:

<img width="934" height="976" alt="Screenshot 2026-08-08 202022" src="https://github.com/user-attachments/assets/483e14a5-9cdd-4b49-aba3-8c72e5a10771" />
