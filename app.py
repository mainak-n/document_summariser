import streamlit as st
import os
import time
import asyncio
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# NEW IMPORT PATH FOR 2026
from langchain_core.documents import Document 
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 1. UI STYLING & CONFIG ---
st.set_page_config(page_title="Gemma 3 Intelligence", layout="wide", page_icon="💎")

# CSS for Modern Dashboard Look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    [data-testid="stSidebar"] { background-color: #1a1c24; color: white; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; border: 1px solid #ddd; padding: 10px; }
    .stChatFloatingInputContainer { background-color: #ffffff; border-top: 1px solid #ddd; }
    h1 { font-family: 'Inter', sans-serif; color: #1a1c24; }
    </style>
""", unsafe_allow_html=True)

def get_or_create_eventloop():
    try: return asyncio.get_event_loop()
    except: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); return loop
get_or_create_eventloop()

# --- 2. SESSION STATE (Optimization Engine) ---
if "messages" not in st.session_state: st.session_state.messages = []
if "brain" not in st.session_state: st.session_state.brain = None
if "indexed_files" not in st.session_state: st.session_state.indexed_files = set()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8a/Google_Gemini_logo.svg", width=50)
    st.title("Gemma 3.0")
    st.markdown("---")
    
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ API Connected")
    else:
        user_key = st.text_input("🔑 API Key", type="password")
        if user_key: os.environ["GOOGLE_API_KEY"] = user_key

    uploaded_files = st.file_uploader("📂 Upload Vault", type="pdf", accept_multiple_files=True)
    
    if st.button("🗑️ Reset Brain"):
        st.session_state.clear()
        st.rerun()

# --- 4. INCREMENTAL PROCESSING LOGIC ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=15))
def safe_append_docs(vector_store, docs):
    vector_store.add_documents(docs)

def process_vault(files):
    # Only process files we haven't seen before
    new_files = [f for f in files if f.name not in st.session_state.indexed_files]
    if not new_files: return

    with st.status("🧠 Evolving Knowledge Base...", expanded=True) as status:
        all_new_docs = []
        for f in new_files:
            st.write(f"📖 Reading {f.name}...")
            reader = PdfReader(f)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            
            # Metadata allows pinpointing sources later
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_new_docs.append(Document(page_content=chunk, metadata={"source": f.name}))
            
            st.session_state.indexed_files.add(f.name)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", transport="rest")
        
        # Initialize or Update
        if st.session_state.brain is None:
            st.session_state.brain = FAISS.from_documents([all_new_docs[0]], embeddings)
            remaining = all_new_docs[1:]
        else:
            remaining = all_new_docs

        if remaining:
            # Incremental addition in small batches to prevent 504 errors
            batch_size = 5
            for i in range(0, len(remaining), batch_size):
                safe_append_docs(st.session_state.brain, remaining[i:i+batch_size])
                time.sleep(0.4)
        
        status.update(label="✅ Knowledge Base Synchronized!", state="complete")

if uploaded_files and os.getenv("GOOGLE_API_KEY"):
    process_vault(uploaded_files)

# --- 5. CHAT ---
st.title("📄 Document Intelligence")
st.caption("Ask anything across your PDF vault. Citations included.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your files..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", transport="rest")
            
            # Retrieval
            docs = st.session_state.brain.similarity_search(prompt, k=5)
            context = ""
            sources = set()
            for d in docs:
                context += f"\n[{d.metadata['source']}]: {d.page_content}\n"
                sources.add(d.metadata['source'])
            
            # Prompt Optimized for Pinpointing
            sys_msg = f"Use the context below to answer. Always name the file you found the answer in.\n\nContext:\n{context}\n\nQuestion: {prompt}"
            response = llm.invoke(sys_msg)
            
            final_text = response.content + f"\n\n---\n**🔍 Cited Sources:** {', '.join(sources)}"
            st.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})
            
        except Exception as e:
            st.error(f"⚠️ Brain Error: {e}")
