# import
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
from dateutil import parser
import dateparser

# Konfigurasi UI
st.set_page_config(page_title="Chatbot Prediksi Harga Beras", layout="centered", page_icon="🌾")

# CSS tampilan chatbot
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
        }
        .chat-title {
            background-color: #2b2b2b;
            color: white;
            text-align: center;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid #555;
        }
        .user-bubble {
            background-color: #3a3a3a;
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 70%;
            margin-left: auto;
            margin-right: 0;
            margin-top: 10px;
            margin-bottom: 10px;
            word-wrap: break-word;
        }
        .bot-bubble {
            background-color: #2e2e2e;
            color: white;
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 70%;
            margin-right: auto;
            margin-left: 0;
            margin-top: 10px;
            margin-bottom: 10px;
            word-wrap: break-word;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-title">Chatbot Prediksi Harga Beras di Jawa Barat</div>', unsafe_allow_html=True)


# Load Dataset dan Model (MENGGUNAKAN LOGIKA BARU)
@st.cache_resource(show_spinner=False)
def load_model_and_data():
    # Load data dan parse tanggal seperti di skrip Colab
    df = pd.read_csv("Prediksi_Data.csv", parse_dates=["Tanggal"])

    # !! PERUBAHAN KUNCI: Menyamakan format tanggal ke MM-DD-YYYY
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True).dt.strftime('%m-%d-%Y')
    df.columns = ['Tanggal', 'Harga_Beras', 'Inflasi', 'Prediksi']

    texts = []
    for _, row in df.iterrows():
        tanggal = row["Tanggal"]
        if pd.notnull(row.get("Harga_Beras")):
            texts.append(f"Tanggal: {tanggal} | Jenis: Harga | Nilai: {row['Harga_Beras']}")
        if pd.notnull(row.get("Inflasi")):
            texts.append(f"Tanggal: {tanggal} | Jenis: Inflasi | Nilai: {row['Inflasi']}")
        if pd.notnull(row.get("Prediksi")):
            texts.append(f"Tanggal: {tanggal} | Jenis: Prediksi | Nilai: {row['Prediksi']}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)

    return df, model, np.array(embeddings), texts

df, model, data_embeddings_np, data_texts = load_model_and_data()


# fungsi embedding
def get_query_embedding(query):
    return model.encode([query])[0]

# FUNGSI NORMALISASI BARU (dari skrip Colab)
def normalize_query_date(query, date_format="%m-%d-%Y"):
    """
    Deteksi apakah query mengandung tanggal dalam format d/m/Y (atau bahasa Indonesia)
    lalu konversi ke format m-d-Y (Bulan-Hari-Tahun).
    """
    date_pattern = r'\b\d{1,2}[\s/-]*[A-Za-z]+[\s/-]*\d{4}\b|\b\d{1,2}[\s/-]\d{1,2}[\s/-]\d{4}\b'
    match = re.search(date_pattern, query, flags=re.IGNORECASE)

    if not match:
        return query  # tidak ada tanggal yang bisa dikenali

    raw_date = match.group(0).strip()

    # Parsing tanggal dengan asumsi dayfirst=True
    parsed_date = dateparser.parse(raw_date, languages=["id", "en"], settings={"DATE_ORDER": "DMY"})
    if not parsed_date:
        try:
            parsed_date = parser.parse(raw_date, dayfirst=True, fuzzy=True)
        except:
            return query  # gagal parsing, return query asli

    # Konversi ke format m-d-Y (SESUAI LOGIKA BARU)
    normalized_date = parsed_date.strftime(date_format)
    print(f"📅 Tanggal terdeteksi: '{raw_date}' → dikonversi ke '{normalized_date}'")

    # Ganti tanggal lama dengan yang baru di query
    query_new = re.sub(re.escape(raw_date), normalized_date, query, flags=re.IGNORECASE)

    return query_new


# FUNGSI SEARCH BARU (dari skrip Colab)
def search(query, data_texts, data_embeddings_np, threshold=0.6, top_k=1):
    q_lower = query.lower()
    results = []
    query = normalize_query_date(query)

    # cek dulu apakah query mengandung tanggal tertentu
    print(f"Pertanyaan: {query}")

    # Regex ini sekarang akan cocok dengan tanggal yang dinormalisasi (MM-DD-YYYY)
    query_date_match = re.search(r"\b\d{1,2}-\d{1,2}-\d{4}\b", query)
    if query_date_match:
      date_parse = query_date_match.group()
      print(date_parse)

    if query_date_match:
        query_date = query_date_match.group(0)  # ambil string tanggal
        print(f"Pertanyaan mengandung tanggal: {query_date}")

        # --- Pastikan format tanggal di df sama (string) ---
        # (ini sudah dipastikan di load_model_and_data)
        if "Tanggal" in df.columns:
            df["Tanggal"] = df["Tanggal"].astype(str).str.strip()

        hasil = df[df["Tanggal"].str.contains(query_date, case=False, na=False)]

        if not hasil.empty:
            for _, row in hasil.iterrows():
                if pd.notnull(row["Harga_Beras"]):
                    results.append((f"Tanggal: {query_date} | Jenis: Harga | Nilai: Rp{row['Harga_Beras']}", 1.0))
                if pd.notnull(row["Inflasi"]):
                    results.append((f"Tanggal: {query_date} | Jenis: Inflasi | Nilai: {row['Inflasi']}%", 1.0))
                if pd.notnull(row["Prediksi"]):
                    results.append((f"Tanggal: {query_date} | Jenis: Prediksi | Nilai: Rp{row['Prediksi']}", 1.0))
            return results
        else:
            # Pesan error yang lebih spesifik dari skrip baru
            return [("⚠️ Maaf, pertanyaan tidak sesuai dengan data historis/prediksi yang ada.", 0.0)]

    # template khusus (dengan teks jawaban dari skrip baru)
    if "faktor" in q_lower:
        return [("Faktor yang mempengaruhi prediksi harga beras adalah "
                 "historis harga beras serta nilai inflasi dalam 4 tahun terakhir.", 1.0)]
    if "hubungan" in q_lower:
        return [("Berdasarkan data yang dimiliki, harga beras dan inflasi "
                 "memiliki hubungan non-linear.", 1.0)]

    # fallback ke embedding search

    if not results:
        # Pesan error yang konsisten dari skrip baru
        return [("⚠️ Maaf, pertanyaan tidak sesuai dengan data historis/prediksi yang ada.", 0.0)]

    return results


# --- Bagian UI Streamlit (Tidak Berubah) ---

# fungsi riwayat chat
if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f'<div class="user-bubble">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg}</div>', unsafe_allow_html=True)


# fungsi input query
query = st.chat_input("Ketik pertanyaan Anda...")

if query:
    # Simpan pesan user
    st.session_state.history.append(("user", query))

    # Jalankan sistem chatbot (menggunakan threshold dari skrip baru)
    results = search(query, data_texts, data_embeddings_np, threshold=0.2, top_k=3)

    # Format jawaban (sedikit dimodifikasi untuk menampilkan semua hasil jika ada)
    if results and results[0][1] == 1.0: # Jika ini adalah pencocokan tanggal (skor 1.0)
        answer = "<br>".join([res[0] for res in results])
    elif results:
        answer = results[0][0] # Ambil hasil semantik terbaik
    else:
        answer = "⚠️ Maaf, saya tidak menemukan jawaban yang sesuai."


    # Simpan jawaban chatbot
    st.session_state.history.append(("bot", answer))

    st.rerun()
