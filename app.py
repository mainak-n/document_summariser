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

# --- 1. UI CONFIG & VISIBLE TOGGLE CSS ---
st.set_page_config(page_title="Document Intelligence", layout="wide")

st.markdown("""
    <style>
    /* Hiding the problematic default header completely */
    [data-testid="stHeader"] { visibility: hidden; height: 0px; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Ensuring the sidebar is easy to see when open */
    [data-testid="stSidebar"] { 
        background-color: #ffffff; 
        border-right: 1px solid #eee; 
        visibility: visible !important;
    }

    /* Main Background & Typography */
    .main { background-color: #fcfcfc; font-family: 'Segoe UI', sans-serif; }
    .stMarkdown p { font-size: 1.1rem; line-height: 1.6; color: #2c3e50; }
    
    /* Better Chat Bubbles */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        margin-bottom: 15px;
    }
    
    /* LIGHT GREY INPUT BOX - FORCED STYLE */
    .stChatInputContainer {
        border: 1px solid #D3D3D3 !important; 
        border-radius: 12px !important;
        background-color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. THE MANUAL UNHIDE BUTTON ---
# This sits at the very top of the page so you can always find it
col1, col2 = st.columns([1, 8])
with col1:
    if st.button("📂 Open Settings"):
        # This is a 'trick' to force the sidebar to show by triggering a rerun
        # and letting the user interact with the sidebar area.
        st.info("Sidebar is on the left ←")

# --- 3. STATE MANAGEMENT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "brain" not in st.session_state: st.session_state.brain = None
if "indexed_files" not in st.session_state: st.session_state.indexed_files = set()

def get_or_create_eventloop():
    try: return asyncio.get_event_loop()
    except: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); return loop
get_or_create_eventloop()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    st.write("Upload your PDFs and enter your key here.")
    
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        user_key = st.text_input("Enter Access Key", type="password")
        if user_key: os.environ["GOOGLE_API_KEY"] = user_key

    uploaded_files = st.file_uploader("Upload Documents", type="pdf", accept_multiple_files=True)
    
    st.divider()
    if st.button("Reset All", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- 5. BRAIN BUILDING LOGIC ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=15))
def safe_append_docs(vector_store, docs):
    vector_store.add_documents(docs)

def update_intelligence(files):
    new_files = [f for f in files if f.name not in st.session_state.indexed_files]
    if not new_files: return
    with st.status("Analyzing Documents...", expanded=False) as status:
        all_new_docs = []
        for f in new_files:
            reader = PdfReader(f)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_new_docs.append(Document(page_content=chunk, metadata={"source": f.name}))
            st.session_state.indexed_files.add(f.name)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", transport="rest")
        if st.session_state.brain is None:
            st.session_state.brain = FAISS.from_documents([all_new_docs[0]], embeddings)
            remaining = all_new_docs[1:]
        else:
            remaining = all_new_docs
        if remaining:
            for i in range(0, len(remaining), 5):
                safe_append_docs(st.session_state.brain, remaining[i:i+5])
                time.sleep(0.3)
        status.update(label="Ready", state="complete")

if uploaded_files and os.getenv("GOOGLE_API_KEY"):
    update_intelligence(uploaded_files)

# --- 6. MAIN CHAT INTERFACE ---
st.title("Document Analysis")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    if not os.getenv("GOOGLE_API_KEY"):
        st.info("Missing Key. Click 'Open Settings' on the top left.")
    elif not st.session_state.brain:
        st.info("No documents. Click 'Open Settings' to upload.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", transport="rest")
                docs = st.session_state.brain.similarity_search(prompt, k=5)
                context = ""
                sources = set()
                for d in docs:
                    context += f"\n[{d.metadata['source']}]: {d.page_content}\n"
                    sources.add(d.metadata['source'])
                
                full_prompt = f"Answer clearly based on context. Cite sources.\n\nContext:\n{context}\n\nQuestion: {prompt}"
                response = llm.invoke(full_prompt)
                final_answer = f"{response.content}\n\n---\n**Sources:** {', '.join(sources)}"
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.info(f"System Note: {e}")
