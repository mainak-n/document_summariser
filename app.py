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

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Document Intelligence", layout="wide")

# Balanced CSS: Clean, High Readability, No Red.
st.markdown("""
    <style>
    /* Main Content Styling */
    .main { background-color: #f9f9fb; }
    
    /* Clean Chat Bubbles */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #eaeaea;
        padding: 15px;
        margin-bottom: 10px;
    }

    /* THE INPUT BOX: Light Grey & Simple */
    .stChatInputContainer {
        border: 1px solid #d1d1d1 !important;
        border-radius: 10px !important;
    }

    /* Text Readability */
    .stMarkdown p {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #333333;
    }

    /* Hide unnecessary Streamlit bits */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

def get_or_create_eventloop():
    try: return asyncio.get_loop()
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
get_or_create_eventloop()

# --- 2. STATE MANAGEMENT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "brain" not in st.session_state: st.session_state.brain = None
if "indexed_files" not in st.session_state: st.session_state.indexed_files = set()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("Control Center")
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        user_key = st.text_input("Access Key", type="password")
        if user_key: os.environ["GOOGLE_API_KEY"] = user_key

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Clear All Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- 4. DATA PROCESSING ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_append(vector_store, docs):
    vector_store.add_documents(docs)

def process_docs(files):
    new_files = [f for f in files if f.name not in st.session_state.indexed_files]
    if not new_files: return

    with st.spinner("Updating Knowledge Base..."):
        all_new_docs = []
        for f in new_files:
            reader = PdfReader(f)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_new_docs.append(Document(page_content=chunk, metadata={"source": f.name}))
            st.session_state.indexed_files.add(f.name)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        if st.session_state.brain is None:
            st.session_state.brain = FAISS.from_documents([all_new_docs[0]], embeddings)
            remaining = all_new_docs[1:]
        else:
            remaining = all_new_docs

        if remaining:
            # Batching prevents the app from hanging
            for i in range(0, len(remaining), 10):
                safe_append(st.session_state.brain, remaining[i:i+10])
                time.sleep(0.1)

if uploaded_files and os.getenv("GOOGLE_API_KEY"):
    process_docs(uploaded_files)

# --- 5. CHAT ---
st.title("Document Analysis")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    if not os.getenv("GOOGLE_API_KEY"):
        st.info("Please provide an API key in the sidebar.")
    elif not st.session_state.brain:
        st.info("Please upload documents to begin.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it")
                
                # Context Retrieval
                docs = st.session_state.brain.similarity_search(prompt, k=4)
                context = ""
                sources = set()
                for d in docs:
                    context += f"\n[File: {d.metadata['source']}]: {d.page_content}\n"
                    sources.add(d.metadata['source'])
                
                response = llm.invoke(f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer precisely and mention the file name.")
                
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.info(f"Notice: {e}")
