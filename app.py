import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import os
import csv
import sys
from datetime import datetime

# --- LIBRARY PENDUKUNG MACHINE LEARNING ---
import sklearn 
from sklearn.preprocessing import FunctionTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from textblob import Word
import __main__

# --- 1. KONFIGURASI HALAMAN (CENTERED) ---
st.set_page_config(
    page_title="Deteksi Phishing AI",
    page_icon="🛡️",
    layout="centered" # Tampilan fokus di tengah
)

# --- 2. KONFIGURASI PATH ---
PATH_MODEL = 'model_phishing_final.pkl' 
PATH_FEEDBACK = 'feedback_dataset.csv'

# --- 3. FUNGSI PREPROCESSING (WAJIB ADA) ---
def preprocessing_teks_lengkap(teks_series):
    if not isinstance(teks_series, pd.Series):
        teks_series = pd.Series(teks_series)
    teks_bersih = teks_series.apply(lambda x: " ".join(w.lower() for w in str(x).split()))
    teks_bersih = teks_bersih.str.replace(r"[^\w\s]", "", regex=True)
    teks_bersih = teks_bersih.str.replace(r"\d", "", regex=True)
    teks_bersih = teks_bersih.apply(lambda x: " ".join(Word(i).lemmatize() for i in x.split()))
    return teks_bersih

# Daftarkan fungsi ke __main__ agar pickle terbaca
setattr(__main__, "preprocessing_teks_lengkap", preprocessing_teks_lengkap)

# --- 4. LOAD MODEL ---
@st.cache_resource
def setup_resources():
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    if os.path.exists(PATH_MODEL):
        try:
            return joblib.load(PATH_MODEL)
        except Exception as e:
            st.error(f"❌ Error Loading Model: {e}")
            return None
    else:
        return None

model = setup_resources()

# --- 5. LOGIKA KEYWORD ---
PHISHING_KEYWORDS = {
    "urgensi": ["urgent", "segera", "immediately", "account suspended", "verify", "24 hours", "action required"],
    "hadiah": ["congratulations", "won", "prize", "lottery", "gift", "cash", "reward", "$", "dollar"],
    "tautan": ["click here", "login", "update", "confirm", "link", "attachment", "secure"],
    "sapaan": ["dear customer", "dear user", "beloved", "sir/madam"]
}

def analisa_keyword_phishing(teks_asli):
    teks_lower = teks_asli.lower()
    temuan = []
    teks_highlight = teks_asli
    found_cats = set()

    for kategori, kata_kunci_list in PHISHING_KEYWORDS.items():
        for kata in kata_kunci_list:
            pattern = re.compile(r'\b' + re.escape(kata) + r'\b', re.IGNORECASE)
            if pattern.search(teks_lower):
                nm = kategori.replace('_', ' ').title()
                temuan.append(f"{nm}: '{kata}'")
                found_cats.add(nm)
                teks_highlight = pattern.sub(
                    f"<span style='background-color:#ffcccc;color:red;font-weight:bold;padding:0 2px;'>{kata}</span>", 
                    teks_highlight
                )
    return teks_highlight, list(set(temuan)), list(found_cats)

def simpan_feedback(teks, ai_label, user_label):
    header = ['Timestamp', 'Teks', 'Prediksi_AI', 'Label_User']
    data = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), teks, ai_label, user_label]
    try:
        exists = os.path.isfile(PATH_FEEDBACK)
        with open(PATH_FEEDBACK, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not exists: w.writerow(header)
            w.writerow(data)
        return True
    except: return False

# --- 6. UI UTAMA (TANPA TABS) ---
st.title("🛡️ Deteksi Phishing AI")
st.markdown("Analisis keamanan email menggunakan **Logistic Regression**.")

if model is None:
    st.warning(f"⚠️ Silakan upload file model **`{PATH_MODEL}`** ke folder aplikasi.")
    st.stop()

# Input Area
st.subheader("1. Masukkan Konten Email")
input_text = st.text_area("", height=200, placeholder="Paste subject dan isi email di sini...")

# Tombol Analisis
if st.button("🔍 Analisa Keamanan", type="primary", use_container_width=True):
    if input_text.strip():
        with st.spinner("Sedang memindai pola bahasa..."):
            try:
                # Prediksi
                pred = model.predict([input_text])[0]
                proba = model.predict_proba([input_text]).max()
                is_phishing = (pred == 1) or (str(pred).lower() in ['phishing', 'spam'])
                
                # Keyword
                hl_text, reasons, cats = analisa_keyword_phishing(input_text)
                
                # Simpan sesi
                st.session_state['hasil'] = {
                    'phishing': is_phishing, 'conf': proba, 
                    'hl': hl_text, 'reasons': reasons, 'cats': cats,
                    'text': input_text
                }
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Teks input tidak boleh kosong.")

# Hasil Analisis
if 'hasil' in st.session_state:
    res = st.session_state['hasil']
    st.divider()
    st.subheader("2. Hasil Analisis")

    # Tampilan Alert
    if res['phishing']:
        st.error(f"🚨 **PERINGATAN: TERINDIKASI PHISHING**\n\nTingkat Keyakinan AI: **{res['conf']:.1%}**")
    else:
        st.success(f"✅ **AMAN (SAFE)**\n\nTingkat Keyakinan AI: **{res['conf']:.1%}**")

    # Detail Indikator
    with st.expander("📝 Lihat Detail & Indikator Kata", expanded=True):
        if res['reasons']:
            st.markdown("**Kata kunci mencurigakan yang ditemukan:**")
            st.markdown(res['hl'], unsafe_allow_html=True)
            st.markdown("---")
            for c in res['cats']:
                st.caption(f"🚩 Kategori: {c}")
        else:
            st.info("Tidak ditemukan kata kunci umum (urgent, hadiah, dll).")
            if res['phishing']:
                st.warning("⚠️ Namun, AI mendeteksi pola struktur kalimat yang mirip dengan data Phishing.")

    # Feedback
    st.divider()
    st.caption("Bantu kami meningkatkan akurasi:")
    col1, col2 = st.columns(2)
    if col1.button("👍 Hasil Benar", use_container_width=True):
        lbl = "Phishing" if res['phishing'] else "Aman"
        simpan_feedback(res['text'], lbl, "Benar")
        st.toast("Terima kasih!")
    
    if col2.button("👎 Hasil Salah", use_container_width=True):
        lbl = "Phishing" if res['phishing'] else "Aman"
        simpan_feedback(res['text'], lbl, "Salah")
        st.toast("Feedback tersimpan.")