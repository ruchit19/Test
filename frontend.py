"""
Website Analyzer Frontend
Enhanced Streamlit frontend with file upload, auto-summary, multi-language, accessibility.
"""

import sys, os, ssl, urllib3, httpx, streamlit as st
from langchain_openai import ChatOpenAI
from main import process_website, process_file, retrieve_context, generate_answer, get_vector_store

# Page config
st.set_page_config(page_title="Website Analyzer", page_icon="🌐", layout="wide")

# SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
http_client = httpx.Client(verify=False)

# API Config
API_KEY = "sk-0HGbLQBcZnn3ErSICkqUCQ"
BASE_URL = "https://genailab.tcs.in"
MODEL_NAME = "azure_ai/genailab-maas-DeepSeek-V3-0324"

@st.cache_resource
def get_llm():
    return ChatOpenAI(base_url=BASE_URL, model=MODEL_NAME, api_key=API_KEY, http_client=http_client, temperature=0.7)

llm = get_llm()

# Session state
if "website_processed" not in st.session_state: st.session_state.website_processed = False
if "current_url" not in st.session_state: st.session_state.current_url = None
if "current_mode" not in st.session_state: st.session_state.current_mode = None
if "messages" not in st.session_state: st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📥 Input Options")
    website_url = st.text_input("Enter Website URL", placeholder="https://example.com")
    uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
    search_mode = st.selectbox("Search Mode", ["Hybrid", "Internal", "External"])
    language = st.selectbox("🌐 Output Language", ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Bengali"])

    if st.button("Process Content", type="primary"):
        if website_url or uploaded_file:
            with st.spinner("Processing..."):
                result = process_file(uploaded_file) if uploaded_file else process_website(website_url)
            if result["status"] == "success":
                st.session_state.website_processed = True
                st.session_state.current_url = website_url or "uploaded_file"
                st.session_state.current_mode = search_mode
                st.session_state.messages = []
                st.success(result["message"])
            else:
                st.error(result["message"])
        else:
            st.error("Please enter a website URL or upload a file")

    if st.button("Reset"):
        st.session_state.website_processed = False
        st.session_state.current_url = None
        st.session_state.current_mode = None
        st.session_state.messages = []

    st.markdown("---")
    st.markdown("### 📋 Status")
    if st.session_state.website_processed:
        st.markdown(f"✅ Processed: `{st.session_state.current_url}`")
    else:
        st.markdown("❌ No content processed yet")

# Main content
if not st.session_state.website_processed:
    st.info("👋 Welcome! Enter a website or upload a file to begin.")
else:
    # Auto-summary after processing
    with st.spinner("Generating summary..."):
        vector_store = get_vector_store()
        if vector_store:
            # Fetch a larger set of chunks without keyword filtering
            context = vector_store.similarity_search("", k=20)
            if context:
                summary = generate_answer("Internal", "Provide a concise summary of the entire content.", context, llm)
                if language != "English":
                    summary = llm.invoke(f"Translate this to {language}: {summary}").content

                st.markdown("### 📄 Summary of Content")
                st.write(summary)
                st.download_button("⬇️ Download Report", summary, file_name="summary.txt")
            else:
                st.warning("No content found to summarize. Try asking a question below.")
        else:
            st.warning("Vector store not available. Please process again.")

    # Chat interface
    st.markdown("### 💬 You are compliant or not? Check here")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = retrieve_context(query, k=4)
                answer = generate_answer(st.session_state.current_mode, query, context, llm)
                if language != "English":
                    answer = llm.invoke(f"Translate this to {language}: {answer}").content
                st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#888;'>Created by NeuralNinjas | Accessible | Multi-language | Dark/Light Mode</div>", unsafe_allow_html=True)
