import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Chat - Gemma 3", layout="wide")
st.title("📄 PDF Chat (Gemma 3 Optimized)")

# --- SIDEBAR & API KEY ---
with st.sidebar:
    st.header("Settings")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key loaded")
    else:
        api_key = st.text_input("Enter Google API Key", type="password")
    
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.rerun()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- FAST PDF PROCESSING ---
if uploaded_file and api_key and st.session_state.vectorstore is None:
    with st.spinner("Analyzing document..."):
        try:
            # Save temp file
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and Split (Larger chunks = fewer API calls = faster)
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            # Modern, faster embedding model
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key
            )
            
            # --- ADAPTIVE LOADING (The Speed Fix) ---
            if len(chunks) <= 15:
                # Small PDF (1-5 pages): Process all at once instantly
                vectorstore = FAISS.from_documents(chunks, embeddings)
            else:
                # Large PDF: Use safety batches
                progress_bar = st.progress(0, text="Processing large document...")
                batch_size = 5
                vectorstore = FAISS.from_documents(chunks[:batch_size], embeddings)
                for i in range(batch_size, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    vectorstore.add_documents(batch)
                    progress_bar.progress(i / len(chunks))
                    time.sleep(2) # Reduced sleep for efficiency
                progress_bar.empty()

            st.session_state.vectorstore = vectorstore
            st.sidebar.success(f"Brain Ready! ({len(chunks)} chunks)")
            st.session_state.messages.append({"role": "assistant", "content": "I've analyzed the PDF. Ask me anything!"})
            
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

# --- CHAT UI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Gemma 3 is thinking..."):
            try:
                # Using the requested Gemma 3 model
                llm = ChatGoogleGenerativeAI(
                    model="gemma-3-27b-it", 
                    google_api_key=api_key, 
                    temperature=0.3,
                    timeout=120
                )
                
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vectorstore.as_retriever(),
                    return_source_documents=True
                )
                
                response = qa_chain.invoke({
                    "question": prompt, 
                    "chat_history": st.session_state.chat_history
                })
                
                answer = response['answer']
                st.session_state.chat_history.append((prompt, answer))
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Chat Error: {e}")
