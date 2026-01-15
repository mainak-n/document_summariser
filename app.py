import streamlit as st
import os
import time
import asyncio
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 1. ASYNC EVENT LOOP FIX ---
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

st.set_page_config(page_title="Gemma 3 Hybrid Intelligence", layout="wide", page_icon="🧠")
st.title("🧠 Multi-PDF Intelligence (Gemma 3)")
get_or_create_eventloop()

# --- 2. SIDEBAR & API KEY SETUP ---
with st.sidebar:
    st.header("Upload Portal")
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key loaded from Secrets.")
    else:
        user_key = st.text_input("Enter Google API Key", type="password")
        if user_key:
            os.environ["GOOGLE_API_KEY"] = user_key

    uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Clear App Cache"):
        st.session_state.clear()
        st.rerun()

# --- 3. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if "brain" not in st.session_state: st.session_state.brain = None
if "mode" not in st.session_state: st.session_state.mode = None
if "full_text" not in st.session_state: st.session_state.full_text = ""

# --- 4. RESILIENT RAG BUILDING LOGIC ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=15))
def safe_add_to_vectorstore(vector_store, chunks):
    """Retries with exponential backoff if Google API hits a 504 timeout."""
    vector_store.add_texts(chunks)

def build_brain(all_text):
    # Smaller chunks help avoid huge payloads that cause 504s
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = text_splitter.split_text(all_text)
    
    # Forced REST transport and high timeout for Streamlit stability
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        transport="rest",
        request_options={"timeout": 1200}
    )
    
    progress_text = "Building AI Brain. Please wait..."
    progress_bar = st.progress(0, text=progress_text)
    
    # Initialize store with the first chunk
    vector_store = FAISS.from_texts([chunks[0]], embeddings)
    
    # Process remaining chunks in small batches
    total = len(chunks)
    for i in range(1, total):
        safe_add_to_vectorstore(vector_store, [chunks[i]])
        progress_bar.progress(i/total, text=f"Processing chunk {i} of {total}...")
        time.sleep(0.5) # Prevent rate-limiting
        
    progress_bar.empty()
    return vector_store

# --- 5. DATA PROCESSING ---
if uploaded_files and os.getenv("GOOGLE_API_KEY") and not st.session_state.brain:
    with st.spinner("Analyzing PDF volume..."):
        all_content = ""
        total_pages = 0
        for pdf in uploaded_files:
            reader = PdfReader(pdf)
            total_pages += len(reader.pages)
            for page in reader.pages:
                all_content += page.extract_text() or ""
        
        # Hybrid Strategy Decision
        if total_pages <= 5:
            st.session_state.mode = "DIRECT"
            st.session_state.full_text = all_content
            st.session_state.brain = "LOADED"
            st.sidebar.info("🚀 Mode: Direct Context (Fastest)")
        else:
            st.session_state.mode = "RAG"
            st.session_state.brain = build_brain(all_content)
            st.sidebar.info("📂 Mode: RAG (Vector Search Active)")

# --- 6. CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about these documents..."):
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("Please provide an API Key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Use Gemma 3 via REST for reliability
                llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", transport="rest")
                
                # Retrieval step
                if st.session_state.mode == "DIRECT":
                    context = st.session_state.full_text
                else:
                    search_results = st.session_state.brain.similarity_search(prompt, k=4)
                    context = "\n\n".join([doc.page_content for doc in search_results])
                
                # Final AI Call
                final_response = llm.invoke(f"CONTEXT FROM PDF:\n{context}\n\nUSER QUESTION: {prompt}")
                
                st.markdown(final_response.content)
                st.session_state.messages.append({"role": "assistant", "content": final_response.content})
            except Exception as e:
                st.error(f"AI Error: {e}")
