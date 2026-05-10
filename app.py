import os
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =========================
# UI
# =========================

st.set_page_config(
    page_title="MFU Research Grant Bot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 BDA_Project2_Group3")

st.markdown("""
### 📌 Project: MFU Research Fund Chatbot

ระบบถาม-ตอบระเบียบทุนวิจัย โดยใช้ Retrieval-Augmented Generation (RAG)

---

### 👥 Group Members (Group 3)

- 6631501003 – Korravee Yimyuan
- 6631501004 – Kittamet Winyayong
- 6631501008 – Kitticheat Suttipipat
- 6631501009 – Kittinan Pinchaisiri
- 6631501011 – Kittiphat Jantho
- 6631501024 – Chirat Sirisrichattra

---
""")

st.caption("ถามตอบระเบียบทุนวิจัยเพื่อพัฒนาการเรียนรู้ มหาวิทยาลัยแม่ฟ้าหลวง")

# =========================
# LOAD RETRIEVER
# =========================

@st.cache_resource(show_spinner="📚 กำลังโหลดเอกสาร...")
def load_retriever():

    docs = []

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    docs += PyPDFDirectoryLoader(DATASET_DIR).load()

    for file in os.listdir(DATASET_DIR):
        if file.endswith(".docx"):
            docs += Docx2txtLoader(os.path.join(DATASET_DIR, file)).load()

    if len(docs) == 0:
        st.error("❌ ไม่พบไฟล์ PDF หรือ DOCX ใน dataset/")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    try:
        bm25 = BM25Retriever.from_documents(splits)
        bm25.k = 6

        retriever = EnsembleRetriever(
            retrievers=[bm25, faiss_retriever],
            weights=[0.5, 0.5]
        )
    except Exception:
        st.warning("⚠️ BM25 ใช้ไม่ได้ → ใช้ FAISS อย่างเดียว")
        retriever = faiss_retriever

    return retriever

# =========================
# GET ANSWER (Retrieval-only)
# =========================

def get_answer(question, retriever):

    expanded_query = f"{question} ทุนวิจัย เบิกจ่าย งวด ค่าตอบแทน"
    docs = retriever.invoke(expanded_query)

    if not docs:
        return "ไม่พบข้อมูลในระเบียบ"

    results = []
    for i, doc in enumerate(docs[:3]):
        content = doc.page_content.strip()
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")

        # แสดง source ถ้ามี
        meta = ""
        if source:
            filename = os.path.basename(source)
            meta = f"📄 **{filename}**"
            if page != "":
                meta += f" หน้า {int(page) + 1}"

        if meta:
            results.append(f"{meta}\n\n{content}")
        else:
            results.append(f"📄 **ข้อมูลที่ {i+1}**\n\n{content}")

    return "\n\n---\n\n".join(results)

# =========================
# INIT
# =========================

retriever = load_retriever()

# =========================
# CHAT UI
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 กำลังค้นหาคำตอบ..."):
            answer = get_answer(prompt, retriever)

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
