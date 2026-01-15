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

# --- 1. CLEAN UI & PERMANENT SIDEBAR ---
st.set_page_config(page_title="Document Intelligence", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Hide Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="sidebar-close"] { display: none !important; }
    
    /* Clean Main Background */
    .main { background-color: #ffffff; font-family: 'Segoe UI', sans-serif; }
    
    /* FLAT CHAT BUBBLES */
    [data-testid="stChatMessage"] {
        background-color: transparent;
        border: none !important;
        box-shadow: none !important;
        margin-bottom: 20px;
    }
    [data-testid="stChatMessage"]::after {
        content: "";
        display: block;
        padding-top: 15px;
        border-bottom: 1px solid #f0f2f6;
    }
    
    /* FONT CONSISTENCY: Force same font even for bold/code outputs */
    .stMarkdown, .stMarkdown p, .stMarkdown span, code, pre {
        font-size: 1.1rem !important; 
        line-height: 1.6 !important; 
        color: #1e293b !important;
        font-family: 'Segoe UI', sans-serif !important;
        font-weight: normal !important;
    }
    
    h1 { font-weight: 800; color: #0f172a; letter-spacing: -1px; }
    
    /* LIGHT GREY INPUT BOX */
    .stChatInputContainer {
        border: 1px solid #e2e8f0 !important; 
        border-radius: 8px !important;
        background-color: #ffffff !important;
    }

    [data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #f1f5f9; }
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

    uploaded_files = st.file_uploader("Upload Documents", type="pdf", accept_multiple_files=True)
    
    st.divider()
    if st.button("Reset System", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- 4. BRAIN BUILDING ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=15))
def safe_append_docs(vector_store, docs):
    vector_store.add_documents(docs)

def update_intelligence(files):
    new_files = [f for f in files if f.name not in st.session_state.indexed_files]
    if not new_files: return

    with st.status("Reading Documents & Creating Vector Database...", expanded=False) as status:
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
        status.update(label="Knowledge Base Updated", state="complete")

if uploaded_files and os.getenv("GOOGLE_API_KEY"):
    update_intelligence(uploaded_files)

# --- 5. MAIN INTERFACE ---
st.title("Document Analysis")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    if not os.getenv("GOOGLE_API_KEY"):
        st.info("Please provide an API key in the sidebar.")
    elif not st.session_state.brain:
        st.info("Please upload documents to begin.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # Placeholder prevents the double-highlight/blurred ghosting effect
            response_placeholder = st.empty()
            
            with st.status("Thinking.......", expanded=False) as status:
                try:
                    llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", transport="rest")
                    
                    docs = st.session_state.brain.similarity_search(prompt, k=5)
                    context = ""
                    for d in docs:
                        context += f"\n[{d.metadata['source']}]: {d.page_content}\n"
                    
                    full_prompt = (
                        f"Provide a professional answer using the context below. "
                        f"List sources at the end. Do not use markdown styling like bolding.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {prompt}"
                    )
                    
                    response = llm.invoke(full_prompt)
                    final_answer = response.content
                    status.update(label="Complete", state="complete")
                    
                    # Update placeholder and session state
                    response_placeholder.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    
                except Exception as e:
                    status.update(label="Error", state="error")
                    st.error(f"Analysis failed: {e}")
