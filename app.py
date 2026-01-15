import streamlit as st
import asyncio
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

# --- THE FIX: EVENT LOOP HELPER ---
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
        raise ex

# --- PAGE CONFIG ---
st.set_page_config(page_title="Direct PDF Chat", page_icon="📄")
st.title("📄 Simple PDF Chat")

# Call the helper at the start
get_or_create_eventloop()

# --- SIDEBAR & API KEY ---
with st.sidebar:
    st.header("Settings")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("Enter Google API Key", type="password")
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CHAT UI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- PROCESSING & CHAT ---
if uploaded_file and api_key:
    # 1. Extract text
    reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()

    # 2. Handle Chat
    if prompt := st.chat_input("Ask about this PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Force the LLM to run in the loop we created
                llm = ChatGoogleGenerativeAI(
                    model="gemma-3-27b-it", 
                    google_api_key=api_key
                )
                
                context_prompt = f"Use this text to answer: {pdf_text}\n\nUser: {prompt}"
                
                # The LLM call is where the error used to happen
                response = llm.invoke(context_prompt)
                
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"AI Error: {e}")
else:
    st.info("Upload a PDF and enter your API Key.")
