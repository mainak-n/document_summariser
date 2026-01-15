import streamlit as st
import os
import time
import asyncio
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 1. CONFIG & UI STYLING ---
st.set_page_config(page_title="Gemma 3 Intelligence", layout="wide", page_icon="💎")

# Custom CSS for a fresh, modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatFloatingInputContainer { background-color: rgba(255,255,255,0.8); backdrop-filter: blur(10px); }
    [data-testid="stSidebar"] { background-image: linear-gradient(#2e3440, #4c566a); color: white; }
    .st-emotion-cache-1c7bg2 { border-radius: 15px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    h1 { color: #1e1e1e; font-weight: 800; letter-spacing: -1px; }
    </style>
""", unsafe_allow_html=True)

def get_or_create_eventloop():
    try: return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        return loop
get_or_create_eventloop()

# --- 2. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "brain" not in st.session_state: st.session_state.brain = None
if "indexed_files" not in st.session_state: st.session_state.indexed_files = set()
if "mode" not in st.session_state: st.session_state.mode = "DIRECT"

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("💎 Gemma v3")
    st.markdown("---")
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        user_key = st.text_input("🔑 Google API Key", type="password")
        if user_key: os.environ["GOOGLE_API_KEY"] = user_key

    uploaded_files = st.file_uploader("📂 Upload PDF Vault", type="pdf", accept_multiple_files=True)
    
    with st.expander("🛠️ Advanced Settings"):
        st.info(f"Active Files: {len(st.session_state.indexed_files)}")
        if st.button("🗑️ Reset All Cache"):
            st.session_state.clear()
            st.rerun()

# --- 4. OPTIMIZED INCREMENTAL PROCESSING ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=15))
def safe_append_docs(vector_store, docs):
    vector_store.add_documents(docs)

def process_new_files(files):
    new_files = [f for f in files if f.name not in st.session_state.indexed_files]
    if not new_files: return
    
    with st.status("🧠 Evolving Intelligence...", expanded=True) as status:
        all_new_docs = []
        for f in new_files:
            st.write(f"Reading {f.name}...")
            reader = PdfReader(f)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_new_docs.append(Document(page_content=chunk, metadata={"source": f.name}))
            st.session_state.indexed_files.add(f.name)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", transport="rest")
        
        if st.session_state.brain is None:
            st.session_state.brain = FAISS.from_documents([all_new_docs[0]], embeddings)
            remaining_docs = all_new_docs[1:]
        else:
            remaining_docs = all_new_docs

        if remaining_docs:
            for i in range(0, len(remaining_docs), 5): # Batch size of 5
                safe_append_docs(st.session_state.brain, remaining_docs[i:i+5])
                time.sleep(0.5)
        
        status.update(label="✅ Brain Updated!", state="complete", expanded=False)

if uploaded_files and os.getenv("GOOGLE_API_KEY"):
    process_new_files(uploaded_files)
    st.session_state.mode = "RAG" if len(st.session_state.indexed_files) > 0 else "DIRECT"

# --- 5. CHAT INTERFACE ---
st.title("📄 Multi-Document Intelligence")
st.caption("Gemma 3.0 | Retrieval-Augmented Generation | Verified Sources")

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question across your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", transport="rest")
            
            # Retrieval with source tracking
            docs = st.session_state.brain.similarity_search(prompt, k=4)
            context = ""
            sources = set()
            for d in docs:
                context += f"\n[{d.metadata['source']}]: {d.page_content}\n"
                sources.add(d.metadata['source'])
            
            # Formulate Response
            sys_prompt = f"Answer using the context. Cite sources clearly. Context:\n{context}\n\nQuestion: {prompt}"
            response = llm.invoke(sys_prompt)
            
            # Build an engaging response with a source list
            full_reply = response.content
            if sources:
                full_reply += f"\n\n---\n**Sources used:** {', '.join(sources)}"
            
            st.markdown(full_reply)
            st.session_state.messages.append({"role": "assistant", "content": full_reply})
            
        except Exception as e:
            st.error(f"⚠️ Intelligence Error: {e}")
