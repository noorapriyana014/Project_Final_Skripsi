import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import os
import csv
import sys
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime

# --- LIBRARY PENDUKUNG MACHINE LEARNING ---
# Penting: Import ini diperlukan agar pickle mengenali struktur data saat di-load
import sklearn 
from sklearn.preprocessing import FunctionTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from textblob import Word
import __main__

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Phishing AI",
    page_icon="🛡️",
    layout="wide" # Layout wide agar visualisasi lebih lega
)

# --- 2. KONFIGURASI PATH ---
# SESUAIKAN DENGAN NAMA FILE DI NOTEBOOK ANDA
PATH_MODEL = 'model_phishing_final.pkl' 
PATH_FEEDBACK = 'feedback_dataset.csv'
PATH_DATASET_VISUAL = 'Phishing_Email.csv' # Opsional: Untuk tab visualisasi

# --- 3. FUNGSI PREPROCESSING (WAJIB ADA & SAMA PERSIS) ---
# Fungsi ini disalin dari notebook Anda agar logika pembersihan datanya konsisten.
def preprocessing_teks_lengkap(teks_series):
    # Konversi ke Series jika inputnya numpy array/list (terjadi saat prediksi)
    if not isinstance(teks_series, pd.Series):
        teks_series = pd.Series(teks_series)

    # Lowercase & Split
    teks_bersih = teks_series.apply(lambda x: " ".join(w.lower() for w in str(x).split()))
    # Hapus Tanda Baca (Regex)
    teks_bersih = teks_bersih.str.replace(r"[^\w\s]", "", regex=True)
    # Hapus Angka
    teks_bersih = teks_bersih.str.replace(r"\d", "", regex=True)
    # Lemmatization (TextBlob)
    teks_bersih = teks_bersih.apply(lambda x: " ".join(Word(i).lemmatize() for i in x.split()))

    return teks_bersih

# --- PENTING: PENDAFTARAN FUNGSI KE __MAIN__ ---
# Trik ini menipu Pickle agar mengira fungsi ini didefinisikan di script utama,
# sama seperti saat Anda menjalankannya di Jupyter Notebook.
setattr(__main__, "preprocessing_teks_lengkap", preprocessing_teks_lengkap)

# --- 4. LOAD MODEL & RESOURCE ---
@st.cache_resource
def setup_resources():
    # Download resource NLTK secara diam-diam
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    # Load Model
    if os.path.exists(PATH_MODEL):
        try:
            loaded_model = joblib.load(PATH_MODEL)
            return loaded_model
        except ModuleNotFoundError as e:
            st.error(f"❌ Error Library: {e}. Pastikan requirements.txt sudah benar.")
            return None
        except Exception as e:
            st.error(f"❌ Error Loading Model: {e}")
            return None
    else:
        return None

model = setup_resources()

# --- 5. DATA KEYWORDS PHISHING (DATABASE) ---
PHISHING_KEYWORDS = {
    "urgensi": ["urgent", "segera", "immediately", "account suspended", "verify", "24 hours", "action required"],
    "hadiah": ["congratulations", "won", "prize", "lottery", "gift", "cash", "reward", "$", "dollar"],
    "tautan_mencurigakan": ["click here", "login", "update", "confirm", "link", "attachment", "secure"],
    "sapaan_umum": ["dear customer", "dear user", "beloved", "sir/madam"]
}

def analisa_keyword_phishing(teks_asli):
    teks_lower = teks_asli.lower()
    temuan = []
    teks_highlight = teks_asli
    found_categories = set()

    for kategori, kata_kunci_list in PHISHING_KEYWORDS.items():
        for kata in kata_kunci_list:
            # Regex \b untuk mencocokkan kata utuh
            pattern = re.compile(r'\b' + re.escape(kata) + r'\b', re.IGNORECASE)
            if pattern.search(teks_lower):
                nama_kat = kategori.replace('_', ' ').title()
                temuan.append(f"{nama_kat}: '{kata}'")
                found_categories.add(nama_kat)
                # Highlight
                teks_highlight = pattern.sub(
                    f"<span style='background-color:#ffcccc;color:red;padding:0 2px;font-weight:bold;'>{kata}</span>", 
                    teks_highlight
                )
    return teks_highlight, list(set(temuan)), list(found_categories)

def simpan_feedback(teks, label_ai, label_user):
    header = ['Timestamp', 'Teks', 'Prediksi_AI', 'Label_User']
    data = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), teks, label_ai, label_user]
    try:
        exists = os.path.isfile(PATH_FEEDBACK)
        with open(PATH_FEEDBACK, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not exists: w.writerow(header)
            w.writerow(data)
        return True
    except: return False

# --- 6. UI APLIKASI (TABS) ---
st.title("🛡️ Deteksi Phishing AI")
st.caption("Model: Logistic Regression + SMOTE + TF-IDF")

# Cek Model Dulu
if model is None:
    st.error(f"⚠️ File model **'{PATH_MODEL}'** tidak ditemukan!")
    st.info("📂 Silakan upload file `.pkl` hasil training ke folder yang sama dengan `app.py`.")
    st.stop() # Hentikan eksekusi jika model tidak ada

tab_deteksi, tab_visual = st.tabs(["🕵️ Deteksi Email", "📊 WordCloud Data"])

# === TAB 1: DETEKSI ===
with tab_deteksi:
    col_input, col_hasil = st.columns([1, 1])
    
    with col_input:
        st.subheader("Input Email")
        input_text = st.text_area("Masukkan konten email:", height=250, placeholder="Paste email content here...")
        
        if st.button("🔍 Analisa Sekarang", type="primary", use_container_width=True):
            if input_text.strip():
                st.session_state['teks_input'] = input_text
                with st.spinner("Memproses..."):
                    try:
                        # 1. Prediksi (Pipeline otomatis menjalankan cleaning -> vectorizer -> model)
                        pred_label = model.predict([input_text])[0]
                        proba = model.predict_proba([input_text]).max()
                        
                        # Label mapping (Sesuaikan 1/0 dengan Phishing/Safe)
                        is_phishing = (pred_label == 1) or (str(pred_label).lower() in ['phishing', 'spam'])
                        
                        # 2. Analisis Keyword
                        hl_text, reasons, cats = analisa_keyword_phishing(input_text)
                        
                        st.session_state['hasil'] = {
                            'is_phishing': is_phishing,
                            'score': proba,
                            'highlight': hl_text,
                            'reasons': reasons,
                            'cats': cats
                        }
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Input kosong!")

    with col_hasil:
        st.subheader("Hasil Analisis")
        if 'hasil' in st.session_state and st.session_state['hasil']:
            res = st.session_state['hasil']
            
            # Kartu Hasil
            if res['is_phishing']:
                st.error(f"🚨 **PHISHING DETECTED**\n\nConfidence: {res['score']:.1%}")
            else:
                st.success(f"✅ **EMAIL AMAN (SAFE)**\n\nConfidence: {res['score']:.1%}")
            
            # Highlight Teks
            with st.expander("📝 Lihat Kata Kunci Berbahaya", expanded=True):
                if res['reasons']:
                    st.markdown(res['highlight'], unsafe_allow_html=True)
                    st.write("---")
                    for c in res['cats']: st.caption(f"🚩 {c}")
                else:
                    st.info("Tidak ditemukan keyword umum, namun struktur kalimat terdeteksi mencurigakan oleh AI.")
            
            # Feedback
            st.write("---")
            st.write("Apakah prediksi ini benar?")
            c1, c2 = st.columns(2)
            if c1.button("👍 Benar"):
                simpan_feedback(st.session_state['teks_input'], "Phishing" if res['is_phishing'] else "Aman", "Sesuai")
                st.toast("Terima kasih!")
            if c2.button("👎 Salah"):
                simpan_feedback(st.session_state['teks_input'], "Phishing" if res['is_phishing'] else "Aman", "Salah")
                st.toast("Feedback disimpan.")

# === TAB 2: VISUALISASI ===
with tab_visual:
    st.header("Visualisasi Kata (WordCloud)")
    
    # Cek apakah user sudah upload dataset CSV
    if os.path.exists(PATH_DATASET_VISUAL):
        if st.button("Generate WordCloud"):
            with st.spinner("Membuat grafik..."):
                try:
                    df = pd.read_csv(PATH_DATASET_VISUAL)
                    # Pastikan nama kolom sesuai dataset Anda (Misal: 'Email Type' dan 'Email Text')
                    # Ubah jika perlu: df['label'] / df['teks']
                    cols_check = [c for c in df.columns if 'type' in c.lower() or 'label' in c.lower()]
                    text_check = [c for c in df.columns if 'text' in c.lower() or 'email' in c.lower()]
                    
                    if cols_check and text_check:
                        col_label = cols_check[0]
                        col_text = text_check[0]
                        
                        phishing_text = " ".join(df[df[col_label].astype(str).str.contains('Phishing', case=False)][col_text].astype(str))
                        safe_text = " ".join(df[df[col_label].astype(str).str.contains('Safe', case=False)][col_text].astype(str))

                        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
                        
                        wc_p = WordCloud(width=400, height=300, background_color='black', colormap='Reds').generate(phishing_text)
                        ax[0].imshow(wc_p)
                        ax[0].set_title("PHISHING Keywords", color='red')
                        ax[0].axis('off')
                        
                        wc_s = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(safe_text)
                        ax[1].imshow(wc_s)
                        ax[1].set_title("SAFE Keywords", color='green')
                        ax[1].axis('off')
                        
                        st.pyplot(fig)
                    else:
                        st.warning("Gagal mendeteksi kolom Label/Text otomatis. Periksa nama kolom CSV.")
                except Exception as e:
                    st.error(f"Gagal memuat visualisasi: {e}")
    else:
        st.info(f"Untuk menampilkan fitur ini, silakan upload file dataset **`{PATH_DATASET_VISUAL}`** ke folder yang sama.")