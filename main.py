"""
Website Analyzer RAG Application
Enhanced backend for scraping websites and processing uploaded files.
"""

import os
import ssl
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import httpx
import urllib3
import tiktoken
from PyPDF2 import PdfReader

# Token cache
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
http_client = httpx.Client(verify=False)

# API Config
API_KEY = "sk-0HGbLQBcZnn3ErSICkqUCQ"
BASE_URL = "https://genailab.tcs.in/"
MODEL_NAME = "azure_ai/genailab-maas-DeepSeek-V3-0324"

# Embeddings
embeddings = OpenAIEmbeddings(
    model="azure/genailab-maas-text-embedding-3-large",
    openai_api_key=API_KEY,
    base_url=BASE_URL,
    http_client=http_client
)

# Globals
vector_store = None
current_url = None

def scrape_website(url: str) -> str:
    """Scrape content from a website."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        raise Exception(f"Error scraping website: {str(e)}")

def load_file(file) -> str:
    """Load text or PDF file content."""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])
    else:
        return file.read().decode("utf-8")

def split_text(text: str, source: str) -> list:
    """Split text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]

def create_vector_store(documents: list, source: str) -> Chroma:
    """Create or load a ChromaDB vector store for a given source."""
    global vector_store
    safe_source = source.replace("https://", "").replace("http://", "").replace("/", "_")
    persist_dir = os.path.join("./chroma_db", safe_source)
    if os.path.exists(persist_dir):
        vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    else:
        vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
        vector_store.persist()
    return vector_store

def process_website(url: str) -> dict:
    """Process a website: scrape, chunk, and store in vector database."""
    global vector_store, current_url
    try:
        text = scrape_website(url)
        if not text or len(text) < 10:
            return {"status": "error", "message": "Not enough content scraped"}
        documents = split_text(text, url)
        vector_store = create_vector_store(documents, url)
        current_url = url
        return {"status": "success", "message": f"Processed {len(documents)} chunks from {url}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def process_file(file) -> dict:
    """Process an uploaded file."""
    global vector_store, current_url
    try:
        text = load_file(file)
        documents = split_text(text, "uploaded_file")
        vector_store = create_vector_store(documents, "uploaded_file")
        current_url = "uploaded_file"
        return {"status": "success", "message": f"Processed {len(documents)} chunks from uploaded file"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def retrieve_context(query: str, k: int = 4) -> list:
    """Retrieve relevant context from the vector store."""
    global vector_store
    if vector_store is None:
        return []
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

def generate_answer(search_mode: str, query: str, context: list, llm) -> str:
    """
    You are a financial regulatory compliance expert, with scope limited strictly to **capital markets** (securities, exchanges, trading, reporting, disclosure, governance, insider trading, market abuse, etc.). 
If the clause is outside capital market compliance, respond: "Outside the scope of capital market compliance."

Task:
1. Classify the compliance clause into one of the following categories:
   - Reporting Obligation
   - Disclosure Requirement
   - Penalty Clause
   - Prohibition
   - Record-Keeping
   - Governance Requirement
   - Audit Requirement

2. Assess compliance risk level:
   - Low
   - Medium
   - High
   - Critical

   Consider:
   - Presence of penalties
   - Short deadlines (<48 hours)
   - Fraud or insider trading
   - Regulatory suspension risk

3. Compare company internal policy against regulatory requirements:
   - Fully met
   - Partially met
   - Not addressed

4. Provide a **Compliance Score (%)** with an RGB indicator:
   - Green (80–100%) → Strong compliance
   - Yellow (50–79%) → Moderate compliance
   - Red (0–49%) → Weak compliance
5. Provide missing compliance details from t   

Instructions:
- Only evaluate clauses relevant to capital market compliance.
- If clause is outside scope, do not classify or score — simply state "Outside the scope of capital market compliance."
- Keep answers concise, structured, and professional.

    """
    context_text = "\n\n".join([doc.page_content for doc in context]) if context else ""
    if search_mode == "Hybrid":
        mode_instructions = "Use both internal document context and external sources."
    elif search_mode == "Internal":
        mode_instructions = "Only rely on internal context."
    elif search_mode == "External":
        mode_instructions = "Ignore internal context and rely only on external sources."
    else:
        mode_instructions = "Default to internal analysis."
    prompt = f"""
Search Mode: {search_mode}
Mode Instructions: {mode_instructions}

Context:
{context_text}

Question: {query}
"""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def get_vector_store():
    global vector_store
    return vector_store
