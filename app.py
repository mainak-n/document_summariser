import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Gemma 3 Server Brain", layout="wide")
st.title("🧠 Persistent PDF Brain (Gemma 3)")

# API Key Retrieval
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("Google API Key", type="password")

# --- 2. THE CORE BRAIN LOGIC ---
def build_brain_if_missing():
    """Checks for existing index or builds it from the 'data' folder in small batches."""
    if not os.path.exists("faiss_index"):
        st.info("🧠 Brain not found! Creating it now...")
        try:
            # Create data folder if missing
            if not os.path.exists("data"):
                os.makedirs("data") 
                st.warning("⚠️ No 'data' folder found. Created empty one. Please add PDFs to it.")
                return

            # Load PDFs
            loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            
            if not documents:
                st.warning("⚠️ No PDFs found in 'data' folder. Brain creation skipped.")
                return

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(documents)
            
            # Initialize Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=API_KEY
            )
            
            # --- STABILITY PATCH: BATCH PROCESSING ---
            # Instead of sending all at once (causes 504), we send 2 at a time.
            batch_size = 2
            progress_bar = st.progress(0, text="Initializing Brain connection...")
            
            # Start the vector store with the first batch
            vector_store = FAISS.from_documents(text_chunks[:batch_size], embeddings)
            time.sleep(1.5) # Give the API a moment
            
            # Add remaining chunks in small batches
            total_chunks = len(text_chunks)
            for i in range(batch_size, total_chunks, batch_size):
                batch = text_chunks[i : i + batch_size]
                vector_store.add_documents(batch)
                
                # Update UI
                progress = min(i / total_chunks, 1.0)
                progress_bar.progress(progress, text=f"Processing chunk {i}/{total_chunks}...")
                time.sleep(1.2) # Small pause to prevent server timeout

            # Save locally for persistence
            vector_store.save_local("faiss_index")
            progress_bar.empty()
            st.success("🎉 Brain created successfully on the server!")
            
        except Exception as e:
            st.error(f"❌ Failed to build brain: {e}")

# --- 3. UI & CHAT LOGIC ---
if API_KEY:
    # Build brain if it doesn't exist on disk
    build_brain_if_missing()

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your server documents..."):
        if os.path.exists("faiss_index"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    # Load the brain
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
                    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    
                    # Gemma 3 Setup
                    llm = ChatGoogleGenerativeAI(
                        model="gemma-3-27b-it", 
                        google_api_key=API_KEY, 
                        temperature=0.3,
                        timeout=120
                    )
                    
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vector_store.as_retriever(),
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
        else:
            st.error("Vector index not found. Please add PDFs to the 'data' folder and restart.")
else:
    st.warning("Please enter your Google API Key to activate the brain.")
