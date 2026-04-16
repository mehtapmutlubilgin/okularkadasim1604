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
    # Klasör isminiz Colab'da neyse onu yazın (Örn: "okul_asistani_v2_db")
    persist_dir = "okul_asistani_v2_db" 
    
    if not os.path.exists(persist_dir):
        st.error(f"Hata: '{persist_dir}' klasörü GitHub deponuzda bulunamadı!")
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Sıfırdan oluşturmak yerine 'persist_directory'den yüklüyoruz
    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vector_db

# --- 2. CEVAP ÜRETME ---
def ask_asistant(v_db, query):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    docs = v_db.similarity_search(query, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    system_msg = "Sen MEB Mevzuat Asistanısın. Yanıtların KISA ve NET olmalı."

    chat = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {query}"}
        ],
        model="llama-3.1-8b-instant",
        temperature=0
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
