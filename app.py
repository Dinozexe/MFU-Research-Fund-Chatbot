import os
import torch
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


@st.cache_resource(show_spinner="กำลังโหลดเอกสาร...")
def load_retriever():
    docs = []
    if os.path.exists("dataset"):
        docs += PyPDFDirectoryLoader("dataset").load()
        for f in os.listdir("dataset"):
            if f.endswith(".docx"):
                docs += Docx2txtLoader(os.path.join("dataset", f)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", "\n", "(?<=\\(\\d\\))", " ", ""]
    ).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 8

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_pipeline():
    torch.cuda.empty_cache()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model_id = "scb10x/llama-3-typhoon-v1.5x-8b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False
    ), tokenizer


def get_answer(message, retriever, pipe, tokenizer):
    expanded_query = (
        f"{message} "
        f"ค่าตอบแทน เบิกจ่าย งวด ทุนสนับสนุน allowance disbursement "
        f"researcher grant installment payment"
    )
    retrieved_docs = retriever.invoke(expanded_query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    messages = [
        {
            "role": "system",
            "content": (
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
        },
        {"role": "user", "content": message}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = pipe(prompt)
    answer = result[0]["generated_text"].strip()

    for stop in ["<|eot_id|>", "<|end_of_text|>", "คำถาม:", "Question:", "Q:", "User:", "Human:", "Assistant:"]:
        if stop in answer:
            answer = answer.split(stop)[0].strip()

    return answer if answer else "ไม่พบข้อมูลในระเบียบ"


st.set_page_config(page_title="MFU Research Grant Bot", page_icon="🎓", layout="centered")
st.title("🎓 MFU Research Grant Bot")
st.caption("ถามตอบระเบียบทุนวิจัยเพื่อพัฒนาการเรียนรู้ มหาวิทยาลัยแม่ฟ้าหลวง")

retriever = load_retriever()
pipe, tokenizer = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาคำตอบ..."):
            answer = get_answer(prompt, retriever, pipe, tokenizer)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
