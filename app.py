import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai

# Page Config
st.set_page_config(page_title="PDF Summarizer & Chat", layout="wide")

st.title("📄 PDF Summarizer & Chat (Gemini Powered)")

# Sidebar for API Key and File Upload
with st.sidebar:
    st.header("Settings")
    
    # Check if API Key is in secrets
    if "GOOGLE_API_KEY" in st.secrets:
        st.success("API Key loaded from secrets")
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("Enter your Google Gemini API Key", type="password")
    st.markdown("[Get your free API key here](https://aistudio.google.com/app/apikey)")
    
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if "chat_history" in st.session_state:
            st.session_state.chat_history = []
        st.rerun()

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process PDF if uploaded
if uploaded_file and api_key and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF... This might take a moment."):
        try:
            # Save temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            
            # Split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # Increased slightly for better context
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            
            # Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            
            # --- FIXED VECTOR STORE CREATION (BATCHED) ---
            try:
                # Optimized Settings
                batch_size = 5    # Send 5 chunks at once (Efficiency)
                sleep_time = 2    # Wait 2s between calls (Rate Limiting)
                
                progress_text = "Creating embeddings (Optimized Mode)..."
                my_bar = st.progress(0, text=progress_text)
                
                # Process first batch to initialize vectorstore
                first_batch = chunks[:batch_size]
                vectorstore = FAISS.from_documents(first_batch, embeddings)
                time.sleep(sleep_time) # Respect rate limits
                
                # Process remaining chunks
                total_chunks = len(chunks)
                
                # Loop starting from the second batch
                for i in range(batch_size, total_chunks, batch_size):
                    # Update progress bar
                    progress = min(i / total_chunks, 1.0)
                    my_bar.progress(progress, text=f"Processing chunk {i}/{total_chunks}")
                    
                    # Create batch
                    batch = chunks[i : i + batch_size]
                    
                    # Retry logic
                    for attempt in range(3):
                        try:
                            vectorstore.add_documents(batch)
                            break
                        except Exception as e:
                            if attempt == 2:
                                st.warning(f"Skipping batch starting at {i} due to error: {e}")
                            time.sleep(2 * (attempt + 1)) # Exponential backoff
                    
                    time.sleep(sleep_time) # Regular rate limit sleep
                
                my_bar.empty()
                st.session_state.vectorstore = vectorstore
                st.sidebar.success("PDF Processed Successfully!")
                
                # Add initial assistant message
                st.session_state.messages.append({"role": "assistant", "content": "I've read your document. What would you like to know?"})
                
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}. Try a smaller PDF.")
                st.session_state.vectorstore = None

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        
        finally:
            # Cleanup temp file
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your PDF..."):
    if not api_key:
        st.error("Please enter your Google API Key in the sidebar.")
    elif st.session_state.vectorstore is None:
        st.error("Please upload a PDF document first.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Added timeout to prevent hanging
                llm = ChatGoogleGenerativeAI(
                    model="gemma-3-27b-it", 
                    google_api_key=api_key, 
                    temperature=0.3,
                    timeout=60
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
                
                # Update chat history for context
                st.session_state.chat_history.append((prompt, answer))
                
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
