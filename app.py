import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from huggingface_hub import InferenceClient

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# ใส่ HF_TOKEN ใน Streamlit Secrets (Settings → Secrets) ชื่อ key ว่า HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN", st.secrets.get("HF_TOKEN", ""))
MODEL_ID  = "scb10x/llama-3-typhoon-v1.5x-8b-instruct"   # เปลี่ยนได้ถ้า HF Inference ไม่รองรับ
DATASET_DIR = "dataset"

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

สังเกต: ค่าตอบแทน (allowance) และ การซื้อครุภัณฑ์ (hardware) เป็นคนละข้อกันในระเบียบ อย่าสับสนระหว่างกัน
"""

# ─── LOAD & BUILD RETRIEVER (cached — runs once per session) ───────────────────
@st.cache_resource(show_spinner="📚 กำลังโหลดเอกสารและสร้าง Retriever...")
def load_retriever():
    if not os.path.exists(DATASET_DIR):
        st.error(f"ไม่พบโฟลเดอร์ '{DATASET_DIR}' — โปรดเพิ่มไฟล์ PDF/DOCX แล้วรีสตาร์ท")
        st.stop()

    docs = []

    # PDF
    pdf_loader = PyPDFDirectoryLoader(DATASET_DIR)
    docs += pdf_loader.load()

    # DOCX
    for fname in os.listdir(DATASET_DIR):
        if fname.endswith(".docx"):
            docs += Docx2txtLoader(os.path.join(DATASET_DIR, fname)).load()

    if not docs:
        st.error("ไม่พบเอกสารใน dataset/ โปรดเพิ่มไฟล์ PDF หรือ DOCX")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", "\n", "(?<=\\(\\d\\))", " ", ""]
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    faiss_ret = vectorstore.as_retriever(search_kwargs={"k": 8})

    bm25_ret = BM25Retriever.from_documents(splits)
    bm25_ret.k = 8

    return EnsembleRetriever(
        retrievers=[bm25_ret, faiss_ret],
        weights=[0.5, 0.5]
    )

# ─── LLM via HuggingFace Inference API ────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 กำลังเชื่อมต่อโมเดล...")
def get_client():
    if not HF_TOKEN:
        st.warning("⚠️ ไม่พบ HF_TOKEN — โปรดตั้งค่าใน Streamlit Secrets หรือ Environment Variable")
    return InferenceClient(token=HF_TOKEN or None)


def ask_llm(client: InferenceClient, message: str, context: str) -> str:
    system_prompt = (
        "คุณคือ AI ตอบคำถามระเบียบทุนวิจัยเพื่อพัฒนาการเรียนรู้ มหาวิทยาลัยแม่ฟ้าหลวง\n\n"
        "กฎเหล็ก:\n"
        "1. ตอบเฉพาะข้อมูลที่มีใน Context ด้านล่างเท่านั้น\n"
        "2. ถ้าไม่มีข้อมูลในระเบียบ ให้ตอบว่า 'ไม่พบข้อมูลในระเบียบ'\n"
        "3. ตอบสั้น กระชับ เป็นภาษาไทย ห้ามแต่งเติม\n"
        "4. ค่าตอบแทน (allowance) และ ครุภัณฑ์ (hardware) เป็นคนละรายการกัน ห้ามสับสน\n"
        "5. อ่าน Context ทุกข้อก่อนตอบ อย่าหยุดที่ข้อแรกที่เจอ\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"Context:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": message},
    ]

    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=400,
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        # ถ้าโมเดล Typhoon ไม่มีใน HF Inference API ให้เปลี่ยนเป็น fallback
        answer = f"เกิดข้อผิดพลาดในการเรียกโมเดล: {e}"

    # ตัด stop tokens ที่อาจติดมา
    for stop in ["<|eot_id|>", "<|end_of_text|>", "คำถาม:", "Question:", "Q:", "User:", "Human:", "Assistant:"]:
        if stop in answer:
            answer = answer.split(stop)[0].strip()

    return answer if answer else "ไม่พบข้อมูลในระเบียบ"


# ─── STREAMLIT UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MFU Research Fund Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 MFU Research Fund Chatbot")
st.caption("ระบบถาม-ตอบระเบียบทุนวิจัยเพื่อพัฒนาการเรียนรู้ มหาวิทยาลัยแม่ฟ้าหลวง")

with st.expander("👥 Group Members (BDA Project 2 — Group 3)", expanded=False):
    st.markdown("""
    | รหัสนักศึกษา | ชื่อ |
    |---|---|
    | 6631501003 | Korravee Yimyuan |
    | 6631501004 | Kittamet Winyayong |
    | 6631501008 | Kitticheat Suttipipat |
    | 6631501009 | Kittinan Pinchaisiri |
    | 6631501011 | Kittiphat Jantho |
    | 6631501024 | Chirat Sirisrichattra |
    """)

# โหลด retriever และ client ครั้งเดียว
retriever = load_retriever()
client    = get_client()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการสนทนา
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# รับ input
if prompt := st.chat_input("ถามเกี่ยวกับระเบียบทุนวิจัย MFU..."):
    # แสดงข้อความผู้ใช้
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve + Generate
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูล..."):
            expanded_query = (
                f"{prompt} "
                "ค่าตอบแทน เบิกจ่าย งวด ทุนสนับสนุน allowance disbursement "
                "researcher grant installment payment"
            )
            retrieved_docs = retriever.invoke(expanded_query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        with st.spinner("กำลังสร้างคำตอบ..."):
            answer = ask_llm(client, prompt, context)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
