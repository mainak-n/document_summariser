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

# --- 1. PREMIUM AESTHETIC CSS ---
st.set_page_config(page_title="Document Intelligence", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="sidebar-close"] { display: none !important; }
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #f9fafb; border-right: 1px solid #f3f4f6; }
    h1 { font-family: 'Inter', sans-serif; font-weight: 800; color: #111827; letter-spacing: -0.05em; }
    .stMarkdown p, .stMarkdown span { font-family: 'Inter', sans-serif; font-size: 1.05rem !important; line-height: 1.7 !important; color: #374151 !important; }
    [data-testid="stChatMessage"] { background-color: transparent !important; border-bottom: 1px solid #f3f4f6 !important; padding: 30px 10px !important; }
    .stChatInputContainer { border: 1px solid #e5e7eb !important; border-radius: 12px !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOGIC & STATE ---
def get_or_create_eventloop():
    try: return asyncio.get_event_loop()
    except: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); return loop
get_or_create_eventloop()

if "messages" not in st.session_state: st.session_state.messages = []
if "brain" not in st.session_state: st.session_state.brain = None
if "indexed_files" not in st.session_state: st.session_state.indexed_files = set()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("### Settings")
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        user_key = st.text_input("Access Key", type="password", placeholder="Enter Google API Key")
        if user_key: os.environ["GOOGLE_API_KEY"] = user_key

    st.markdown("---")
    uploaded_files = st.file_uploader("Library", type="pdf", accept_multiple_files=True)
    
    if st.button("Purge Brain", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- 4. ENGINE (RAG WITH PAGE METADATA) ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=15))
def safe_append_docs(vector_store, docs):
    vector_store.add_documents(docs)

def update_intelligence(files):
    new_files = [f for f in files if f.name not in st.session_state.indexed_files]
    if not new_files: return

    with st.status("Syncing library with page mapping...", expanded=False) as status:
        all_new_docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        for f in new_files:
            reader = PdfReader(f)
            # Process page by page to capture correct numbers
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    chunks = splitter.split_text(page_text)
                    for chunk in chunks:
                        all_new_docs.append(Document(
                            page_content=chunk, 
                            metadata={"source": f.name, "page": i + 1}
                        ))
            st.session_state.indexed_files.add(f.name)

        if not all_new_docs: return
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", transport="rest")
        
        if st.session_state.brain is None:
            st.session_state.brain = FAISS.from_documents([all_new_docs[0]], embeddings)
            remaining = all_new_docs[1:]
        else:
            remaining = all_new_docs

        if remaining:
            for i in range(0, len(remaining), 10):
                safe_append_docs(st.session_state.brain, remaining[i:i+10])
                time.sleep(0.5)
        status.update(label="Library Synced", state="complete")

if uploaded_files and os.getenv("GOOGLE_API_KEY"):
    update_intelligence(uploaded_files)

# --- 5. INTERFACE ---
st.title("Document Analysis")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Inquire about your documents..."):
    if not os.getenv("GOOGLE_API_KEY"):
        st.info("Missing credentials.")
    elif not st.session_state.brain:
        st.info("Library empty.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            res_area = st.empty()
            with st.status("Searching context...", expanded=False) as status:
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", transport="rest")
                    docs = st.session_state.brain.similarity_search(prompt, k=6)
                    
                    context = ""
                    sources_found = []
                    for d in docs:
                        source_info = f"{d.metadata['source']} (Page {d.metadata['page']})"
                        context += f"\n[{source_info}]: {d.page_content}\n"
                        sources_found.append(source_info)
                    
                    full_prompt = (
                        f"Provide a professional, plain-text answer using the context provided. "
                        f"You must include page numbers in your citations.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Query: {prompt}"
                    )
                    
                    response = llm.invoke(full_prompt)
                    ans = response.content
                    status.update(label="Complete", state="complete")
                    
                    res_area.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    status.update(label="System Error", state="error")
                    st.error(f"Error: {e}")
