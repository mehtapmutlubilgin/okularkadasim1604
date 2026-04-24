__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="OKUL ARKADAŞIM", page_icon="🏛️", layout="wide")

# --- CUSTOM CSS (Görsel Tasarım İçin) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTitle { color: white; text-align: center; font-size: 3rem !important; margin-bottom: 20px; }
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
st.markdown("<h1 class='stTitle'>🏛️OKUL ARKADAŞIM</h1>", unsafe_allow_html=True)

# --- SECRETS KONTROLÜ ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("Hata: GROQ_API_KEY Streamlit Secrets panelinde bulunamadı!")
    st.stop()

# --- HIZLI SORULAR (GÖRSEL KARTLAR) ---
st.markdown("### 💡 Hızlı Sorular")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""<div class="card card-red"><h3>📜 Kayıt & Disiplin</h3><ul>
    <li>• Disiplin cezaları nelerdir?</li><li>•Kayıt işlemleri nasıl yapılır?</li>
    <li>• "Kınama" cezası alan öğrencinin dosyasına işlenir mi?</li></ul></div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="card card-blue"><h3>⌛ Devamsızlık</h3><ul>
    <li>• 10/30 gün kuralı nedir?</li><li>• Ortalamam 75 ve toplam 8 gün devamsızlığım var, belge alabilir miyim?</li>
    <li>• 11 gün özürsüz devamsızlığım var, sınıfta kalır mıyım?</li></ul></div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="card card-green"><h3>🎓 Başarı </h3><ul>
    <li>• Teşekkür belgesi alabilmek için ortalamamın kaç olması lazım?</li><li>• Yıl sonu başarı ortalamam 48, sınıfı geçebilir miyim?</li>
    <li>• 86 puan ortalamam var, devamsızlığım olsa da Takdir alabilir miyim?</li></ul></div>""", unsafe_allow_html=True)

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

# --- 2. CEVAP ÜRETME ---
def ask_asistant(v_db, query):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    docs = v_db.similarity_search(query, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    # Colab'daki başarılı kural seti
    system_msg = """Sen MEB Mevzuat Asistanısın. Yanıtların ÇOK KISA (en fazla 2 cümle) ve NET olmalı.
     KESİN ANALİZ KURALLARI (BU SIRAYLA UYGULA):
    1. SINIF GEÇME: Bir öğrencinin geçmesi için İKİ ŞART AYNI ANDA sağlanmalıdır: 
       - Yıl sonu başarı ortalaması en az 50.00 olmalı.
       - Zayıf (başarısız) ders sayısı en fazla 3 olmalı. 
       *Eğer zayıf sayısı 4 veya daha fazlaysa, ortalama kaç olursa olsun öğrenci KALIR.*

    2. DEVAMSIZLIK: 
       - Özürsüz sınır 10 gündür. 10.5 veya 11 gün olan kesin KALIR.
       - Özürlü (raporlu) devamsızlık tek başına kalma sebebi değildir, toplam sınır 30 gündür.

    3. BELGE: Teşekkür 70-84.99, Takdir 85.00+ gerektirir. Devamsızlık artık belge almaya engel değildir.

    4. GÜVENLİK: Eğer soru anlamsız harflerden (asdf, hjh vb.) oluşuyorsa: "Lütfen MEB mevzuatı ile ilgili anlamlı bir soru sorunuz." de.
   
    CEVAP FORMATI:
    - Eğer öğrenci kalıyorsa cevaba "MAALESEF" veya "HAYIR" kelimesiyle başlama; doğrudan durumu açıkla.
    - ÖRNEK: "4 zayıfın olduğu için sınıf tekrarına kalırsın." 
    - ÖRNEK: "Ortalaman 50'nin altında olduğu için geçemezsin."

    TALİMAT: Cevap vermeden önce zayıf sayısını ve devamsızlık türünü kontrol et. Asla uydurma soru sorma."""

    

    chat = client.chat.completions.create(
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {query}"}],
        model="llama-3.1-8b-instant",
        temperature=0
    )
    return chat.choices[0].message.content

# --- SOHBET AKIŞI ---
v_db = load_existing_vector_db()
if v_db:
    # --- YAN MENÜYE SIFIRLA BUTONU EKLEME ---
    with st.sidebar:
        st.markdown("### ⚙️ Ayarlar")
        if st.button("🔄 Sohbeti Sıfırla"):
            st.session_state.messages = []
            st.rerun()
        st.info("Yeni bir öğrenci senaryosu denemeden önce sohbeti sıfırlamanız önerilir.")

    # Session state kontrolü (Mevcut kodunuz buradan devam eder...)
    if "messages" not in st.session_state:
        st.session_state.messages = []
