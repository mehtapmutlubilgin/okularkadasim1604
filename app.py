import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SAYFA YAPISI ---
st.set_page_config(page_title="Okul Asistanı v2", page_icon="🤖")
st.title("🤖 Akıllı Okul Mevzuat Asistanı")

# --- SECRETS KONTROLÜ ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("Hata: GROQ_API_KEY Streamlit Secrets panelinde bulunamadı!")
    st.stop()

# --- 1. KAYITLI VEKTÖR DOSYASINI YÜKLEME ---
@st.cache_resource
def load_existing_vector_db():
    persist_dir = "okul_asistani_v2_db" 
    
    if not os.path.exists(persist_dir):
        st.error(f"Hata: '{persist_dir}' klasörü GitHub deponuzda bulunamadı!")
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vector_db

# --- 2. CEVAP ÜRETME (COLAB MANTĞI İLE GÜNCELLENDİ) ---
def ask_asistant(v_db, query):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    # Colab ile aynı şekilde k=5 bilgisini çekiyoruz
    docs = v_db.similarity_search(query, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    # COLAB'DAKİ TAM SİSTEM MESAJI
    system_msg = """Sen MEB Mevzuat Asistanısın. Yanıtların ÇOK KISA (en fazla 2 cümle) ve NET olmalı.

    ASLA DEĞİŞMEZ ANALİZ KURALLARI:
    1. SORUMLULUK: Sorumluluk sınavı geçme puanı 50'dir. Sınavlar dönemlerin ilk iki haftasında yapılır.
    2. MANTIK: 8 sayısı 10'dan küçüktür; 8 gün devamsızlıkla kalınmaz.
    3. GÜNCEL: Devamsızlık artık başarı belgesi (Takdir/Teşekkür) almaya engel DEĞİLDİR.
    4. RAPOR: Hastane raporları 'Özürlü' devamsızlıktır, 10 günlük özürsüz sınırını etkilemez.
    5. SINIF GEÇME: Ortalaması 50 olsa bile 3 dersten fazla zayıfı (4, 5, 6...) olan öğrenci KALIR.
    6. BELGE: Teşekkür 70-84.99, Takdir 85.00 ve üzeri ortalama gerektirir.
    7. Sorumluluk sınavı geçme notu 50'dir.
    8. Eğer öğrenci 50 ve üzerinde bir not almışsa (50, 60, 70, 80 vb.) KESİNLİKE GEÇMİŞTİR.
    9. Bağlamda (CSV'de) çelişkili rakamlar görürsen (70 gibi), her zaman '50 ve üzeri geçer' kuralını uygula.

    TALİMAT: Sadece sorunun cevabını ver. Gereksiz açıklama ve kuralları tekrar etme."""

    chat = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {query}"}
        ],
        model="llama-3.1-8b-instant",
        temperature=0 # Sıfır risk: Colab ile aynı
    )
    return chat.choices[0].message.content

# --- ANA DÖNGÜ ---
v_db = load_existing_vector_db()

if v_db:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = ask_asistant(v_db, prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
