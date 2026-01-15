import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="Gemma 3 PDF Chat", layout="wide")
st.title("📄 PDF Chat: Max Stability Mode")

# API Key handling
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Google API Key", type="password")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- STABLE BRAIN CREATION ---
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file and api_key and st.session_state.vectorstore is None:
    with st.spinner("Building Brain (Slow & Stable Mode to avoid 504)..."):
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()
            
            # Smaller chunks are easier for the API to process without timing out
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            chunks = text_splitter.split_documents(pages)
            
            # 1. Setup Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key
            )

            # 2. Build FAISS index in tiny batches
            # This prevents the "Deadline Exceeded" error
            batch_size = 2 
            vectorstore = FAISS.from_documents(chunks[:batch_size], embeddings)
            
            prog_bar = st.progress(0)
            for i in range(batch_size, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                vectorstore.add_documents(batch)
                prog_bar.progress(i / len(chunks))
                time.sleep(2) # Mandatory pause to prevent 504 errors
            
            st.session_state.vectorstore = vectorstore
            st.sidebar.success("Brain Created! Gemma 3 is now active.")
            st.rerun()

        except Exception as e:
            st.error(f"Brain Building Error: {e}")
        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

# --- GEMMA 3 CHAT ---
if st.session_state.vectorstore:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask Gemma 3 about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Now we call Gemma 3
                llm = ChatGoogleGenerativeAI(
                    model="gemma-3-27b-it", 
                    google_api_key=api_key, 
                    temperature=0.3,
                    timeout=120
                )
                
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                
                res = qa_chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
                
                answer = res["answer"]
                st.session_state.chat_history.append((prompt, answer))
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Gemma Error: {e}")
