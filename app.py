import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Gemma 3 PDF Portal", layout="wide")
st.title("🧠 Persistent PDF Brain (Gemma 3)")

if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("Google API Key", type="password")

# --- 2. THE CORE BRAIN LOGIC ---
def build_brain():
    """Builds the brain from scratch using files in the 'data' folder."""
    if not os.path.exists("data") or not os.listdir("data"):
        st.warning("📁 No PDFs found in the server's data folder.")
        return

    try:
        st.info("Creating Brain from server files...")
        loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
        
        # Stability Patch: Tiny batches to avoid 504 Deadline Exceeded
        batch_size = 2
        vector_store = FAISS.from_documents(text_chunks[:batch_size], embeddings)
        
        for i in range(batch_size, len(text_chunks), batch_size):
            vector_store.add_documents(text_chunks[i : i + batch_size])
            time.sleep(1.5) 

        vector_store.save_local("faiss_index")
        st.success("🎉 Brain updated successfully!")
    except Exception as e:
        st.error(f"❌ Failed to build brain: {e}")

# --- 3. SIDEBAR UPLOAD PORTAL ---
with st.sidebar:
    st.header("Upload Portal")
    uploaded_file = st.file_uploader("Add new PDF to Brain", type="pdf")
    
    if uploaded_file and API_KEY:
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Save the uploaded file to the 'data' folder
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Saved {uploaded_file.name} to server.")
        
        # Trigger brain rebuild
        if st.button("Update AI Brain"):
            build_brain()
            st.rerun()

    if st.button("Force Rebuild Brain"):
        build_brain()

# --- 4. CHAT LOGIC ---
if API_KEY:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your PDF collection..."):
        if os.path.exists("faiss_index"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
                    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    
                    llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", google_api_key=API_KEY, timeout=120)
                    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
                    
                    response = qa_chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
                    answer = response['answer']
                    
                    st.session_state.chat_history.append((prompt, answer))
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Chat Error: {e}")
        else:
            st.error("AI Brain not yet created. Upload a PDF and click 'Update AI Brain'.")
