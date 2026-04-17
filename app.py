__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MEB Yönetmelik Asistanı", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS (Görsel Tasarım İçin) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTitle { color: white; text-align: center; font-size: 3rem !important; margin-bottom: 20px; }
    
    /* Kart Stilleri */
    .card {
        background-color: #1a1c24;
        border-radius: 15px;
        padding: 20px;
        height: 250px;
        border-top: 5px solid;
        margin-bottom: 20px;
    }
    .card-red { border-color: #ff4b4b; }
    .card-blue { border-color: #0083ff; }
    .card-green { border-color: #00d488; }
    .card h3 { color: white; margin-bottom: 15px; font-size: 1.2rem; }
    .card ul { color: #a3a8b4; list-style-type: none; padding: 0; font-size: 0.9rem; }
    .card li { margin-bottom: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- BAŞLIK ---
st.markdown("<h1 class='stTitle'>🏛️ MEB Yönetmelik Asistanı</h1>", unsafe_allow_html=True)

# --- SECRETS KONTROLÜ ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("Hata: GROQ_API_KEY Streamlit Secrets panelinde bulunamadı!")
    st.stop()

# --- HIZLI SORULAR (GÖRSEL KARTLAR) ---
st.markdown("### 💡 Hızlı Sorular")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""<div class="card card-red"><h3>📜 Kayıt & Disiplin</h3><ul>
    <li>• Disiplin cezaları nelerdir?</li><li>• Kopya cezası nedir?</li>
    <li>• "Kınama" cezası alan öğrencinin dosyasına işlenir mi?</li></ul></div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="card card-blue"><h3>⌛ Devamsızlık</h3><ul>
    <li>• 10/30 gün kuralı nedir?</li><li>• Yarım gün izin devamsızlık sayılır mı?</li>
    <li>• Toplam devamsızlık sınırı ne zaman 60 güne çıkar?</li></ul></div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="card card-green"><h3>🎓 Başarı & Nakil</h3><ul>
    <li>• Kaç zayıfla kalınır?</li><li>• Nakil dönemi ne zamandır?</li>
    <li>• Onur Belgesi alma şartları nelerdir?</li></ul></div>""", unsafe_allow_html=True)

st.markdown("---")

# --- 1. KAYITLI VEKTÖR DOSYASINI YÜKLEME ---
@st.cache_resource
def load_existing_vector_db():
    persist_dir = "okul_asistani_v2_db" 
    if not os.path.exists(persist_dir):
        st.error(f"Hata: '{persist_dir}' klasörü GitHub deponuzda bulunamadı!")
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# --- 2. CEVAP ÜRETME (COLAB MANTIĞI İLE BİREBİR AYNI) ---
def ask_asistant(v_db, query):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    docs = v_db.similarity_search(query, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    # Colab'daki tam sistem mesajı
    system_msg = """Sen MEB Mevzuat Asistanısın. Yanıtların ÇOK KISA (en fazla 2 cümle) ve NET olmalı.
    ASLA DEĞİŞMEZ ANALİZ KURALLARI:
    1. SORUMLULUK: Sorumluluk sınavı geçme puanı 50'dir. Sınavlar dönemlerin ilk iki haftasında yapılır.
    2. MANTIK: 8 sayısı 10'dan küçüktür; 8 gün devamsızlıkla kalınmaz.
    3. GÜNCEL: Devamsızlık artık başarı belgesi (Takdir/Teşekkür) almaya engel DEĞİLDİR.
    4. RAPOR: Hastane raporları 'Özürlü' devamsızlıktır, 10 günlük özürsüz sınırını etkilemez.
    5. SINIF GEÇME: Ortalaması 50 olsa bile 3 dersten fazla zayıfı (4, 5, 6...) olan öğrenci KALIR.
    6. BELGE: Teşekkür 70-84.99, Takdir 85.00 ve üzeri ortalama gerektirir.
    8. Eğer öğrenci 50 ve üzerinde bir not almışsa KESİNLİKE GEÇMİŞTİR.
    9. Bağlamda çelişkili rakamlar görürsen, her zaman '50 ve üzeri geçer' kuralını uygula.
    TALİMAT: Sadece sorunun cevabını ver. Gereksiz açıklama yapma."""

    chat = client.chat.completions.create(
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {query}"}],
        model="llama-3.1-8b-instant",
        temperature=0 # Sıfır risk: Colab ile aynı
    )
    return chat.choices[0].message.content

# --- SOHBET AKIŞI ---
v_db = load_existing_vector_db()

if v_db:
    if "messages
