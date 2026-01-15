import os
import requests
import time
from flask import Flask, request
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
app = Flask(__name__)

# --- CONFIGURATION ---
API_KEY = os.environ.get("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
WEBHOOK_URL = os.environ.get("RENDER_EXTERNAL_URL") 

# --- STABILITY-FOCUSED BRAIN FUNCTION ---
def build_brain_if_missing():
    if not os.path.exists("faiss_index"):
        print("🧠 Brain not found! Creating it now...")
        try:
            if not os.path.exists("data"):
                os.makedirs("data") 
                print("⚠️ No data folder found. Created empty one.")
                return

            loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            
            if not documents:
                print("⚠️ No PDFs found in data folder.")
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(documents)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)

            # --- STABILITY FIX: BATCHED EMBEDDING ---
            batch_size = 5  # Small batches to avoid 504 Deadline Exceeded
            sleep_time = 2  # Pause to avoid Rate Limits
            
            print(f"Total chunks to process: {len(text_chunks)}")
            
            # Initialize Vector Store with the first batch
            vector_store = FAISS.from_documents(text_chunks[:batch_size], embeddings)
            time.sleep(sleep_time)

            # Process remaining batches
            for i in range(batch_size, len(text_chunks), batch_size):
                batch = text_chunks[i : i + batch_size]
                
                # Retry Logic for network spikes
                for attempt in range(3):
                    try:
                        vector_store.add_documents(batch)
                        print(f"✅ Processed {i + len(batch)}/{len(text_chunks)}...")
                        break
                    except Exception as e:
                        if attempt == 2: raise e
                        print(f"⚠️ Retrying batch due to: {e}")
                        time.sleep(5 * (attempt + 1))
                
                time.sleep(sleep_time)

            vector_store.save_local("faiss_index")
            print("🎉 Brain created successfully on the server!")
            
        except Exception as e:
            print(f"❌ Error creating brain: {e}")
    else:
        print("🧠 Brain already exists.")

# Run builder on startup
build_brain_if_missing()

# --- AI SETUP ---
def get_ai_response(user_text):
    try:
        if not API_KEY:
            return "Error: Google API Key is missing."

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
        
        # Load the index
        if not os.path.exists("faiss_index"):
            return "My brain is empty. Please upload PDFs to the 'data' folder."
            
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_text, k=3)
        
        # Use high timeout for the LLM response
        llm = ChatGoogleGenerativeAI(
            model="gemma-3-27b-it", # Flash is faster and more stable for bots
            google_api_key=API_KEY, 
            temperature=0.3,
            timeout=120, # Increased timeout
            convert_system_message_to_human=True
        )
        
        chain = load_qa_chain(llm, chain_type="stuff")
        return chain.run(input_documents=docs, question=user_text)
        
    except Exception as e:
        print(f"AI Error: {e}")
        return "I encountered an error connecting to my brain."

# --- ROUTES ---
@app.route("/", methods=["GET"])
def index():
    return "Telegram Bot is Running! 🚀"

@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def telegram_webhook():
    update = request.get_json()
    if "message" in update and "text" in update["message"]:
        chat_id = update["message"]["chat"]["id"]
        user_text = update["message"]["text"]
        
        answer = get_ai_response(user_text)

        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": chat_id, "text": answer})
    return "OK", 200

@app.route("/set_webhook", methods=["GET"])
def set_webhook():
    webhook_endpoint = f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}"
    telegram_api = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
    response = requests.post(telegram_api, json={"url": webhook_endpoint})
    return f"Webhook setup result: {response.text}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
