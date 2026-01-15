import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Chat - Stability Mode", layout="wide")
st.title("📄 PDF Summarizer & Chat")

# --- API KEY & SESSION STATE ---
with st.sidebar:
    st.header("Settings")
    # Priority: 1. Manual Input, 2. Streamlit Secrets
    user_api_key = st.text_input("Enter Google API Key", type="password")
    
    if user_api_key:
        api_key = user_api_key
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = None
        st.warning("Please provide an API Key.")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.rerun()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- PDF PROCESSING ---
if uploaded_file and api_key and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF with High-Stability Mode..."):
        try:
            # 1. Save File Locally
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Load and Split
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            # 3. Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=api_key
            )
            
            # 4. Batched Vector Store Creation (Fixed for 504 Errors)
            batch_size = 2  # Small batches = less chance of timeout
            sleep_time = 3  # Gap between requests to stay under 60 RPM
            
            progress_bar = st.progress(0, text="Initializing Embeddings...")
            
            # Initialize with first batch
            vectorstore = FAISS.from_documents(chunks[:batch_size], embeddings)
            time.sleep(sleep_time)
            
            # Add remaining chunks
            total_chunks = len(chunks)
            for i in range(batch_size, total_chunks, batch_size):
                batch = chunks[i : i + batch_size]
                
                # Retry loop for stability
                for attempt in range(3):
                    try:
                        vectorstore.add_documents(batch)
                        break
                    except Exception as e:
                        if attempt == 2: raise e
                        time.sleep(5 * (attempt + 1)) # Wait longer on failure
                
                progress_bar.progress(i / total_chunks, text=f"Processed {i}/{total_chunks} chunks...")
                time.sleep(sleep_time)
            
            st.session_state.vectorstore = vectorstore
            progress_bar.empty()
            st.success("PDF Ready!")
            
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

# --- CHAT INTERFACE ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your PDF..."):
    if not api_key or not st.session_state.vectorstore:
        st.error("Please ensure API Key and PDF are loaded.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # High timeout to prevent 504 on long answers
                llm = ChatGoogleGenerativeAI(
                    model="gemma-3-27b-it", 
                    google_api_key=api_key, 
                    temperature=0.3,
                    timeout=120 
                )
                
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(),
                    return_source_documents=False
                )
                
                res = qa_chain.invoke({
                    "question": prompt, 
                    "chat_history": st.session_state.chat_history
                })
                
                answer = res['answer']
                st.write(answer)
                st.session_state.chat_history.append((prompt, answer))
                st.session_state.messages.append({"role": "assistant", "content": answer})
