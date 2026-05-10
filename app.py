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
    AutoTokenizer
)

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_CONTEXT_LENGTH = 3000

FEW_SHOT_EXAMPLES = """
ตัวอย่างการตอบคำถามที่ถูกต้อง:

Q: อาจารย์จะตั้งงบวิจัยอย่างไร
A: อาจารย์สามารถตั้งงบวิจัยได้ตามค่าใช้จ่ายที่คาดว่าจะเกิดขึ้นจริงในโครงการ โดยอ้างอิงรายการค่าใช้จ่ายตามระเบียบของมหาวิทยาลัย

Q: ผู้วิจัยจะได้รับเงินเมื่อไหร่
A: การเบิกจ่ายเงินทุนวิจัยฯ จะแบ่งจ่ายเป็น 3 งวด ได้แก่
งวดที่ 1 50%
งวดที่ 2 30%
งวดที่ 3 20%

Q: ผู้วิจัยสามารถเบิกค่าตอบแทนได้หรือไม่
A: สามารถเบิกได้ไม่เกิน 3,000 บาท ต่อโครงการ
"""

# =========================
# UI (คงของเดิม)
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

    # 🔥 embedding เบา
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(splits, embeddings)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 🔥 BM25 fallback
    try:
        bm25 = BM25Retriever.from_documents(splits)
        bm25.k = 6

        retriever = EnsembleRetriever(
            retrievers=[bm25, faiss_retriever],
            weights=[0.5, 0.5]
        )
    except Exception as e:
        st.warning("⚠️ BM25 ใช้ไม่ได้ → ใช้ FAISS อย่างเดียว")
        retriever = faiss_retriever

    return retriever

# =========================
# LOAD MODEL
# =========================

@st.cache_resource(show_spinner="🤖 กำลังโหลดโมเดล...")
def load_model():
    try:
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
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=False
        )

        return pipe, tokenizer

    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        st.stop()

# =========================
# GENERATE ANSWER
# =========================

def get_answer(question, retriever, pipe, tokenizer):

    expanded_query = f"{question} ทุนวิจัย เบิกจ่าย งวด ค่าตอบแทน"

    docs = retriever.invoke(expanded_query)

    context = "\n\n".join([d.page_content for d in docs])
    context = context[:MAX_CONTEXT_LENGTH]

    system_prompt = f"""
คุณคือ AI ตอบคำถามระเบียบทุนวิจัย

กฎ:
1. ตอบเฉพาะข้อมูลใน Context
2. ถ้าไม่มี → "ไม่พบข้อมูลในระเบียบ"
3. ตอบสั้น กระชับ ภาษาไทย
4. ห้ามแต่งข้อมูล

{FEW_SHOT_EXAMPLES}

Context:
{context}
"""

    # 🔥 safe chat template
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    except:
        prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"

    result = pipe(prompt)

    answer = result[0]["generated_text"].strip()

    # clean output
    stop_tokens = ["Q:", "User:", "Assistant:"]
    for t in stop_tokens:
        if t in answer:
            answer = answer.split(t)[0]

    if not answer:
        answer = "ไม่พบข้อมูลในระเบียบ"

    return answer

# =========================
# INIT
# =========================

retriever = load_retriever()
pipe, tokenizer = load_model()

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
            answer = get_answer(prompt, retriever, pipe, tokenizer)

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
