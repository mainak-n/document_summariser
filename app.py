import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="Simple Gemma Chat", layout="wide")
st.title("📄 Simple PDF Chat (Direct Context)")

# --- API KEY ---
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    API_KEY = st.sidebar.text_input("Google API Key", type="password")

# --- PDF TEXT EXTRACTION ---
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --- SIDEBAR UPLOAD ---
with st.sidebar:
    st.header("Upload Portal")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if st.button("Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()

# --- SESSION STATE ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- CHAT LOGIC ---
if uploaded_file and API_KEY:
    # 1. Extract text once and keep it in memory
    if "pdf_context" not in st.session_state:
        with st.spinner("Reading PDF..."):
            st.session_state.pdf_context = get_pdf_text(uploaded_file)
            st.success("PDF Loaded!")

    # 2. Display Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. User Input
    if prompt := st.chat_input("Ask a question about this PDF..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 4. Get AI Response
        with st.chat_message("assistant"):
            with st.spinner("Gemma 3 is reading and answering..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemma-3-27b-it",
                        google_api_key=API_KEY,
                        temperature=0.3
                    )
                    
                    # Create the context-aware prompt
                    full_prompt = f"""
                    You are a helpful assistant. Use the following PDF content to answer the user's question.
                    
                    PDF CONTENT:
                    {st.session_state.pdf_context}
                    
                    USER QUESTION:
                    {prompt}
                    """
                    
                    response = llm.invoke(full_prompt)
                    answer = response.content
                    
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"AI Error: {e}")
else:
    if not API_KEY:
        st.info("Please enter your API Key in the sidebar.")
    elif not uploaded_file:
        st.info("Please upload a PDF to start the chat.")
