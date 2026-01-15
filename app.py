def update_intelligence(files):
    new_files = [f for f in files if f.name not in st.session_state.indexed_files]
    if not new_files: return

    with st.status("Syncing library...", expanded=False) as status:
        all_new_docs = []
        for f in new_files:
            reader = PdfReader(f)
            # TRACK PAGE NUMBERS MANUALLY
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                page_num = i + 1  # 1-based indexing for citations
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_text(page_text)
                
                for chunk in chunks:
                    all_new_docs.append(Document(
                        page_content=chunk, 
                        metadata={"source": f.name, "page": page_num}
                    ))
            
            st.session_state.indexed_files.add(f.name)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", transport="rest")
        
        if st.session_state.brain is None:
            st.session_state.brain = FAISS.from_documents([all_new_docs[0]], embeddings)
            remaining = all_new_docs[1:]
        else:
            remaining = all_new_docs

        if remaining:
            for i in range(0, len(remaining), 5):
                safe_append_docs(st.session_state.brain, remaining[i:i+5])
                time.sleep(0.3)
        status.update(label="Library Synced", state="complete")

# --- PROMPT LOGIC UPDATE (Inside assistant block) ---
# Ensure context includes page numbers for the LLM to see
context = ""
for d in docs:
    context += f"\n[Source: {d.metadata['source']} | Page: {d.metadata['page']}]: {d.page_content}\n"

full_prompt = (
    f"Provide a sophisticated, professional answer based on the context. "
    f"List citations at the end including file names and specific page numbers. Use plain text.\n\n"
    f"Context:\n{context}\n\n"
    f"Query: {prompt}"
)
