import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Summarizer & Chat", layout="wide")
st.title("📄 PDF Summarizer & Chat (Gemini Powered)")

# --- SIDEBAR & API KEY ---
with st.sidebar:
    st.header("Settings")
    
    # Priority: 1. Secrets 2. User Input
    if "GOOGLE_API_KEY" in st.secrets:
        st.success("API Key loaded from secrets")
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("Enter your Google Gemini API Key", type="password")
    
    st.markdown("[Get your free API key here](https://aistudio.google.com/app/apikey)")
    
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.rerun()

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- PDF PROCESSING LOGIC ---
if uploaded_file and api_key and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF (Resilient Mode)..."):
        try:
            # Save temp file
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and Split
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            # Use text-embedding-004 for better performance
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key,
                task_type="retrieval_document"
            )
            
            # --- STABILITY PARAMETERS ---
            batch_size = 2  # Small batches to prevent 504 Deadline Exceeded
            sleep_time = 3  # Gap to stay under rate limits
            
            progress_bar = st.progress(0, text="Initializing Secure Connection...")

            # 1. Initialize Vector Store with first batch (Multiple Retries)
            vectorstore = None
            for attempt in range(5):
                try:
                    vectorstore = FAISS.from_documents(chunks[:batch_size], embeddings)
                    break
                except Exception as e:
                    if attempt == 4: raise e
                    time.sleep(10 * (attempt + 1)) # Exponential backoff for initial connection

            time.sleep(sleep_time)
            
            # 2. Add remaining chunks
            total_chunks = len(chunks)
            for i in range(batch_size, total_chunks, batch_size):
                batch = chunks[i : i + batch_size]
                
                # Retry loop for adding documents
                for attempt in range(3):
                    try:
                        vectorstore.add_documents(batch)
                        break
                    except Exception as e:
                        if attempt == 2: st.warning(f"Skipped chunk {i} due to API lag.")
                        time.sleep(5 * (attempt + 1))
                
                progress_bar.progress(i / total_chunks, text=f"Learning chunk {i}/{total_chunks}...")
                time.sleep(sleep_time)

            progress_bar.empty()
            st.session_state.vectorstore = vectorstore
            st.sidebar.success("PDF Ready for Chat!")
            st.session_state.messages.append({"role": "assistant", "content": "I've read your document. What would you like to know?"})
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    if not api_key:
        st.error("Please enter your API Key in the sidebar.")
    elif st.session_state.vectorstore is None:
        st.error("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Using Gemma-3 or Gemini-1.5-Flash for high speed
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
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Chat Error: {e}")
