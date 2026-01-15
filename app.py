import streamlit as st
import asyncio
import os
import time
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. SETUP & EVENT LOOP ---
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

st.set_page_config(page_title="Gemma Hybrid Multi-PDF", layout="wide")
st.title("PDF Summariser")
get_or_create_eventloop()

# --- 2. SIDEBAR CONFIG ---
with st.sidebar:
    st.header("Settings")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("Enter Google API Key", type="password")
    
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Clear Memory"):
        st.session_state.messages = []
        st.session_state.full_text = ""
        st.session_state.vector_store = None
        st.session_state.mode = None
        st.rerun()

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "mode" not in st.session_state:
    st.session_state.mode = None

# --- 4. HYBRID PROCESSING LOGIC ---
if uploaded_files and api_key and not st.session_state.mode:
    with st.spinner("Analyzing Documents..."):
        all_text = ""
        total_pages = 0
        all_docs = []
        
        # Extract text and count pages
        for uploaded_file in uploaded_files:
            reader = PdfReader(uploaded_file)
            total_pages += len(reader.pages)
            for page in reader.pages:
                all_text += page.extract_text()
        
        # Decide Mode: Direct vs RAG
        if total_pages <= 5:
            st.session_state.mode = "DIRECT"
            st.session_state.full_text = all_text
            st.sidebar.success(f"Mode: Direct Context ({total_pages} pages)")
        else:
            st.session_state.mode = "RAG"
            st.sidebar.info(f"Mode: RAG Optimized ({total_pages} pages)")
            
            # Split for RAG
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_text(all_text)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
            
            # Stable Batching for FAISS (Avoids 504 Errors)
            batch_size = 2
            vector_store = FAISS.from_texts(chunks[:batch_size], embeddings)
            for i in range(batch_size, len(chunks), batch_size):
                vector_store.add_texts(chunks[i:i+batch_size])
                time.sleep(1) # API Safety Gap
            
            st.session_state.vector_store = vector_store

# --- 5. CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    if not api_key:
        st.error("Please enter an API Key.")
    elif not uploaded_files:
        st.info("Please upload PDFs first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", google_api_key=api_key)
                
                if st.session_state.mode == "DIRECT":
                    context = st.session_state.full_text
                else:
                    # RAG Retrieval
                    docs = st.session_state.vector_store.similarity_search(prompt, k=5)
                    context = "\n\n".join([doc.page_content for doc in docs])
                
                final_prompt = f"""
                Context from PDFs:
                {context}
                
                Question: {prompt}
                
                Instructions: Answer based ONLY on the context provided. If not mentioned, say you don't know.
                """
                
                response = llm.invoke(final_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
            except Exception as e:
                st.error(f"AI Error: {e}")
