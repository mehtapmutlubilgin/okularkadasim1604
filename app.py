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
    docs = v_db.similarity_search(query, k=3)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    system_msg = """Sen MEB Mevzuat Uzmanısın. Sadece doğrudan cevabı ver, asla giriş cümlesi kurma.

    1. SINIF GEÇME ANALİZİ (MATEMATİKSEL ÖNCELİK):
       - Eğer Ortalama >= 50.00 ise: "Ortalaman 50 barajının üzerinde olduğu için (zayıf sayın 3 veya daha azsa) sınıfı geçersin."
       - Eğer Ortalama < 50.00 ise: "Ortalaman 50 barajının altında olduğu için zayıf sayına bakılmaksızın sınıf tekrarına kalırsın."
    
    2. ZAYIF SAYISI KONTROLÜ:
       - Zayıf Sayısı >= 4 ise: "4 veya daha fazla zayıfın olduğu için ortalaman kaç olursa olsun sınıf tekrarına kalırsın."
       - Zayıf Sayısı 2 veya 3 + Ortalama >= 50 ise: "3 zayıfa kadar Madde 58 uyarınca sorumlu olarak sınıfı geçersin."

    3. DEVAMSIZLIK ANALİZİ:
       - Özürsüz <= 10 ve Toplam <= 30 ise: "Devamsızlık sınırını aşmadığın için kalmazsın."
       - Özürsüz > 10 veya Toplam > 30 ise: "Devamsızlık sınırını aştığın için sınıf tekrarına kalırsın."

    4. BELGELER VE GÜNCEL KURALLAR:
       - Teşekkür: 70.00 - 84.99 | Takdir: 85.00 ve üzeri.
       - ÖNEMLİ: Devamsızlık artık belge (Takdir/Teşekkür) almaya engel DEĞİLDİR.
       - BAĞLAM YASAĞI: Sınıf geçme defteri, sınav kağıdı saklama gibi alakasız verileri asla cevaba ekleme.

    YASAKLAR: 
    - "Maalesef", "Kontrol edelim", "Evet/Hayır" gibi girişler yapma. 
    - Kullanıcının verdiği rakamı (Örn: 65) mutlaka 50 ile kıyasla. 65 sayısı 50'den büyüktür, bu yüzden doğrudan 'geçersin' de.
    - Sadece tek cümlelik net hüküm ver."""
    
    chat = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {query}"}
        ],
        model="llama-3.1-8b-instant",
        temperature=0 # Cevapların değişmemesi için sıfırda tutuyoruz
    )
    return chat.choices[0].message.content

    chat = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Bağlam verileri: {baglam}\n\nKullanıcı Sorusu: {query}"}
        ],
        model="llama-3.1-8b-instant",
        temperature=0
    )
    return chat.choices[0].message.content

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
# --- MESAJLARI GÖRÜNTÜLEME VE YENİ SORU ALMA ---
    # Eski mesajları ekrana bas
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Yeni soru girişi
    if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
        # Kullanıcı mesajını kaydet ve ekrana bas
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistan cevabı üret ve ekrana bas
        with st.chat_message("assistant"):
            with st.spinner("Mevzuat inceleniyor..."): # Kullanıcı beklerken hoş bir yükleme simgesi
                response = ask_asistant(v_db, prompt)
                st.markdown(response)
        
        # Asistan cevabını hafızaya kaydet
        st.session_state.messages.append({"role": "assistant", "content": response})
