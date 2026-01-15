import streamlit as st
import os
import time
import asyncio
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 1. CLEAN UI (NO BORDERS) & PERMANENT SIDEBAR ---
st.set_page_config(page_title="Document Intelligence", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="sidebar-close"] { display: none !important; }
    
    /* Main Background */
    .main { background-color: #ffffff; font-family: 'Segoe UI', sans-serif; }
    
    /* CLEAN CHAT: No Borders, No Shadows */
    [data-testid="stChatMessage"] {
        background-color: transparent;
        border: none !important;
        box-shadow: none !important;
        padding-left: 0px;
        padding-right: 0px;
        margin-bottom: 20px;
    }
    
    /* Divider for readability since borders are gone */
    [data-testid="stChatMessage"]::after {
        content: "";
        display: block;
        margin-top: 15px;
        border-bottom: 1px solid #f0f2f6;
    }

    /* Typography */
    .stMarkdown p { font-size: 1.1rem; line-height: 1.7; color: #1e293b; }
    h1 { font-weight: 800; color: #0f172a; letter-spacing: -1px; }
    
    /* CLEAN INPUT: Light Grey and Minimal */
    .stChatInputContainer { 
        border: 1px solid #e2e8f0 !important; 
        border-radius: 8px !important;
        background-color: #ffffff !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
        background-color: #f8fafc; 
        border-right: 1px solid #f1f5f9; 
    }
    </style>
""", unsafe_allow_html=True)

def get_or_create_eventloop():
    try: return asyncio.get_event_loop()
    except: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); return loop
get_or_create_eventloop()

# --- 2. STATE MANAGEMENT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "brain" not in st.session_state: st.session_state.brain = None
if "indexed_files" not in st.session_state: st.session_state.indexed_files = set()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        user_key = st.text_input("Access Key", type="password", placeholder="Enter Google API Key")
        if user_key: os.environ["GOOGLE_API_KEY"] = user_key

    uploaded_files = st.file_uploader("Upload (6+ Pages)", type="pdf", accept_multiple_files=True)
    
    if st.button("Reset System", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- 4. BRAIN BUILDING (CREATING VECTOR DB) ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=15))
def safe_append_docs(vector_store, docs):
    vector_store.add_documents(docs)

def update_intelligence(files):
    new_files = [f for f in files if f.name not in st.session_state.indexed_files]
    if not new_files: return

    for f in new_files:
        reader = PdfReader(f)
        if len(reader.pages) <= 5:
            st.sidebar.info(f"Skipped {f.name} (Requires 6+ pages)")
            st.session_state.indexed_files.add(f.name)
            continue

        with st.status(f"Creating Vector Database for {f.name}...", expanded=False) as status:
            text = "".join([p.extract_text() or "" for p in reader.pages])
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_text(text)
            docs_to_add = [Document(page_content=chunk, metadata={"source": f.name}) for chunk in chunks]
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", transport="rest")
            
            if st.session_state.brain is None:
                st.session_state.brain = FAISS.from_documents([docs_to_add[0]], embeddings)
                remaining = docs_to_add[1:]
            else:
                remaining = docs_to_add

            if remaining:
                for i in range(0, len(remaining), 5):
                    safe_append_docs(st.session_state.brain, remaining[i:i+5])
                    time.sleep(0.3)
            
            st.session_state.indexed_files.add(f.name)
            status.update(label="Database Synced", state="complete")

if uploaded_files and os.getenv("GOOGLE_API_KEY"):
    update_intelligence(uploaded_files)

# --- 5. MAIN INTERFACE ---
st.title("Document Analysis")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("How can I help you today?"):
    if not os.getenv("GOOGLE_API_KEY"):
        st.info("Please enter your Access Key in the sidebar.")
    elif not st.session_state.brain:
        st.info("Please upload a document with more than 5 pages.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # Thinking State
            with st.status("Thinking.......", expanded=False) as status:
                try:
                    llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", transport="rest")
                    docs = st.session_state.brain.similarity_search(prompt, k=5)
                    context = "".join([f"\n[{d.metadata['source']}]: {d.page_content}\n" for d in docs])
                    
                    response = llm.invoke(f"Context:\n{context}\n\nQuestion: {prompt}")
                    answer = response.content
                    status.update(label="Complete", state="complete")
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    status.update(label="Error", state="error")
                    st.error(f"Failed to process: {e}")
