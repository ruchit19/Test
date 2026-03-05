import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile
import os

# ----------------------------
# Configuration
# ----------------------------

os.environ["TIKTOKEN_CACHE_DIR"] = "./token"

BASE_URL = "https://genailab.tcs.in"
API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------
# LLM Setup
# ----------------------------

llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-0HGbLQBcZnn3ErSICkqUCQ",
    temperature=0.3
)

embedding_model = OpenAIEmbeddings(
    base_url=BASE_URL,
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-0HGbLQBcZnn3ErSICkqUCQ"
)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="RAG PDF Summarizer")
st.title("📄 RAG-powered PDF Summarizer")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # ----------------------------
    # Step 1: Extract Text
    # ----------------------------
    raw_text = extract_text(temp_file_path)

    if not raw_text.strip():
        st.error("No readable text found in PDF.")
        st.stop()

    # ----------------------------
    # Step 2: Chunking
    # ----------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(raw_text)

    # ----------------------------
    # Step 3: Create Vector Store
    # ----------------------------
    with st.spinner("Indexing document..."):
        vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            persist_directory="./chroma_index"
        )
        vectordb.persist()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # ----------------------------
    # Step 4: RAG Chain
    # ----------------------------
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retri
