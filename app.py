import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Brain Chat", layout="wide")
st.title("🧠 PDF Chat (Stability & Persistence Mode)")

# --- CONFIGURATION & SECRETS ---
# Look for GOOGLE_API_KEY in Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")

INDEX_PATH = "faiss_index"

# --- FUNCTIONS ---

def build_vector_store(uploaded_file):
    """Processes PDF in small batches to avoid 504 Deadline Exceeded."""
    try:
        # Save temp file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and Split
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
        
        # --- BATCHED PROCESSING ---
        batch_size = 3  # Small batches for stability
        sleep_time = 2  # Pause for rate limits
        
        progress_bar = st.progress(0, text="Creating Brain...")
        
        # Initialize with first batch
        vector_store = FAISS.from_documents(chunks[:batch_size], embeddings)
        time.sleep(sleep_time)
        
        # Loop through remaining batches
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            # Retry logic
            for attempt in range(3):
                try:
                    vector_store.add_documents(batch)
                    break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(5)
            
            progress_bar.progress(i / len(chunks), text=f"Chunk {i}/{len(chunks)} processed...")
            time.sleep(sleep_time)
        
        # Save locally like your reference code
        vector_store.save_local(INDEX_PATH)
        progress_bar.empty()
        return vector_store

    finally:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("Build/Reset Brain"):
        if uploaded_file and API_KEY:
            with st.spinner("Processing..."):
                st.session_state.vector_store = build_vector_store(uploaded_file)
                st.success("Brain created and saved!")
        else:
            st.error("Missing File or API Key")

# --- CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question about your PDF..."):
    if not API_KEY:
        st.error("Please add your API Key")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI Response
        with st.chat_message("assistant"):
            try:
                # Load Embeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
                
                # Load Local Index (Safe check)
                if os.path.exists(INDEX_PATH):
                    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                    docs = vector_store.similarity_search(prompt, k=4)
                    
                    # LLM Setup from your reference code
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash", 
                        google_api_key=API_KEY, 
                        temperature=0.3,
                        timeout=120,
                        convert_system_message_to_human=True
                    )
                    
                    # Load QA Chain from your reference code
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=prompt)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.warning("Please upload a PDF and click 'Build Brain' first.")
            
            except Exception as e:
                st.error(f"AI Error: {e}")
