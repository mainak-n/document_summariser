import streamlit as st
import os
import time
import asyncio
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 1. SETUP ---
st.set_page_config(page_title="Gemma 3 Resilience Chat", layout="wide")
st.title("🛡️ Anti-Timeout PDF Chat (Gemma 3)")

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

get_or_create_eventloop()

# --- 2. THE API KEY HACK ---
# Setting the key as an environment variable is much more stable than passing it to the class.
with st.sidebar:
    st.header("Setup")
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        user_key = st.text_input("Enter Google API Key", type="password")
        if user_key:
            os.environ["GOOGLE_API_KEY"] = user_key

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Reset Everything"):
        st.session_state.clear()
        st.rerun()

# --- 3. THE RESILIENCE LOGIC ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_embed_chunks(vector_store, chunks):
    """Retries embedding with exponential backoff if a 504 occurs."""
    vector_store.add_texts(chunks)

def build_resilient_brain(all_text):
    # Smaller chunks = smaller payloads = less likely to time out
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_text(all_text)
    
    # Initialize Embeddings using REST and a very long timeout
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        transport="rest",
        request_options={"timeout": 2000} 
    )
    
    progress_bar = st.progress(0, "Initializing Brain...")
    
    # Start with the first chunk to create the store
    vector_store = FAISS.from_texts([chunks[0]], embeddings)
    
    # Add remaining chunks ONE BY ONE with retry logic
    total = len(chunks)
    for i in range(1, total):
        try:
            safe_embed_chunks(vector_store, [chunks[i]])
            progress_bar.progress(i/total, f"Embedding chunk {i}/{total}...")
        except Exception as e:
            st.error(f"Critical Failure at chunk {i}: {e}")
            return None
            
    progress_bar.empty()
    return vector_store

# --- 4. MAIN APP LOGIC ---
if "messages" not in st.session_state: st.session_state.messages = []
if "brain" not in st.session_state: st.session_state.brain = None

if uploaded_files and os.getenv("GOOGLE_API_KEY") and not st.session_state.brain:
    with st.spinner("Building Brain... This may take a minute but it won't crash."):
        full_text = ""
        total_pages = 0
        for pdf in uploaded_files:
            reader = PdfReader(pdf)
            total_pages += len(reader.pages)
            for page in reader.pages:
                full_text += page.extract_text() or ""
        
        if total_pages <= 5:
            st.session_state.mode = "DIRECT"
            st.session_state.full_text = full_text
            st.session_state.brain = "READY"
        else:
            st.session_state.mode = "RAG"
            st.session_state.brain = build_resilient_brain(full_text)

# --- 5. CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", transport="rest")
            
            if st.session_state.mode == "DIRECT":
                context = st.session_state.full_text
            else:
                docs = st.session_state.brain.similarity_search(prompt, k=4)
                context = "\n\n".join([d.page_content for d in docs])
            
            response = llm.invoke(f"Context: {context}\n\nUser Question: {prompt}")
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
        except Exception as e:
            st.error(f"Chat Error: {e}")
