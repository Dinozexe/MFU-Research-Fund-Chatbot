import os
import torch
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

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_ID = "google/gemma-2-2b-it"

FEW_SHOT_EXAMPLES = """
ตัวอย่างการตอบคำถามที่ถูกต้อง:

Q: อาจารย์จะตั้งงบวิจัยอย่างไร
A: อาจารย์สามารถตั้งงบวิจัยได้ตามค่าใช้จ่ายที่คาดว่าจะเกิดขึ้นจริงในโครงการ โดยอ้างอิงรายการค่าใช้จ่ายตามระเบียบของมหาวิทยาลัย

Q: ผู้วิจัยจะได้รับเงินเมื่อไหร่
A: การเบิกจ่ายเงินทุนวิจัยฯ จะแบ่งจ่ายเป็น 3 งวด ได้แก่
งวดที่ 1 เบิกจ่ายร้อยละ 50 ของจำนวนเงินทุนที่ได้รับสนับสนุน ภายใน 30 วันหลังจากทำสัญญารับทุน
งวดที่ 2 เบิกจ่ายร้อยละ 30 ของจำนวนเงินทุนที่ได้รับสนับสนุน หลังจากส่งรายงานความก้าวหน้าวิจัย
งวดที่ 3 เบิกจ่ายร้อยละ 20 ของจำนวนเงินทุนที่ได้รับสนับสนุน หลังจากส่งรายงานวิจัยฉบับสมบูรณ์

Q: ผู้วิจัยสามารถเบิกค่าตอบแทนได้หรือไม่
A: สามารถเบิกจ่ายค่าตอบแทนผู้วิจัยและผู้ช่วยวิจัย (ถ้ามี) ในจำนวนรวมกันไม่เกิน 3,000 บาท ต่อโครงการ

สังเกต:
ค่าตอบแทน (allowance) และ การซื้อครุภัณฑ์ (hardware)
เป็นคนละข้อกันในระเบียบ อย่าสับสนระหว่างกัน
"""

# =========================
# PAGE CONFIG
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

    # PDF
    pdf_loader = PyPDFDirectoryLoader(DATASET_DIR)
    docs += pdf_loader.load()

    # DOCX
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".docx"):
            path = os.path.join(DATASET_DIR, file)
            docs += Docx2txtLoader(path).load()

    if len(docs) == 0:
        st.error("❌ ไม่พบไฟล์ PDF หรือ DOCX ในโฟลเดอร์ dataset")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            "(?<=\\(\\d\\))",
            " ",
            ""
        ]
    )

    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 8

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble

# =========================
# LOAD MODEL
# =========================

@st.cache_resource(show_spinner="🤖 กำลังโหลดโมเดล AI (ขนาดเล็ก)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32, 
        device_map="cpu"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=250, 
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
        return_full_text=False
    )

    return pipe, tokenizer

# =========================
# GENERATE ANSWER
# =========================

def get_answer(question, retriever, pipe, tokenizer):

    expanded_query = (
        f"{question} "
        f"ค่าตอบแทน เบิกจ่าย งวด ทุนสนับสนุน "
        f"allowance disbursement "
        f"researcher grant installment payment"
    )

    retrieved_docs = retriever.invoke(expanded_query)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "คุณคือ AI ตอบคำถามระเบียบทุนวิจัย "
                "เพื่อพัฒนาการเรียนรู้ มหาวิทยาลัยแม่ฟ้าหลวง\n\n"

                "กฎเหล็ก:\n"
                "1. ตอบเฉพาะข้อมูลที่มีใน Context เท่านั้น\n"
                "2. ถ้าไม่มีข้อมูล ให้ตอบว่า 'ไม่พบข้อมูลในระเบียบ'\n"
                "3. ตอบสั้น กระชับ เป็นภาษาไทย\n"
                "4. ห้ามแต่งเติมข้อมูล\n"
                "5. ค่าตอบแทน และ ครุภัณฑ์ เป็นคนละรายการ\n"
                "6. อ่าน Context ทุกข้อก่อนตอบ\n\n"

                f"{FEW_SHOT_EXAMPLES}\n\n"
                f"Context:\n{context}"
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    result = pipe(prompt)

    answer = result[0]["generated_text"].strip()

    stop_tokens = [
        "<|eot_id|>",
        "<|end_of_text|>",
        "Q:",
        "Question:",
        "User:",
        "Assistant:",
        "Human:"
    ]

    for token in stop_tokens:
        if token in answer:
            answer = answer.split(token)[0].strip()

    if not answer:
        answer = "ไม่พบข้อมูลในระเบียบ"

    return answer

# =========================
# LOAD SYSTEM
# =========================

retriever = load_retriever()
pipe, tokenizer = load_model()

# =========================
# CHAT HISTORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# USER INPUT
# =========================

if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("🔍 กำลังค้นหาคำตอบ..."):

            answer = get_answer(
                prompt,
                retriever,
                pipe,
                tokenizer
            )

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
