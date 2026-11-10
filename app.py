import hashlib

import streamlit as st
from pypdf import PdfReader

from rag.chunker import chunk_pages
from rag.retriever import Retriever
from router import classify_query
from research.orchestrator import run


st.title("🔎 DeepScout AI")
st.write("Your AI-powered knowledge and research assistant.")


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "documents" not in st.session_state:
    st.session_state.documents = []  # List of dicts: {name, hash, chunks_count}

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None


def render_pdf_sources(pdf_sources):

    if not pdf_sources:
        return

    st.markdown("### 📚 PDF Sources")

    # Group by (source, page) and keep the highest score for each unique source page
    unique_sources = {}
    
    for result in pdf_sources:
        source = result.get("source", "Unknown")
        page = result.get("page", "Unknown")
        score = result.get("score", 0)
        
        # Create a key for (source, page)
        key = (source, page)
        
        # Keep only the highest score for this source-page combination
        if key not in unique_sources or score > unique_sources[key]["score"]:
            unique_sources[key] = {
                "source": source,
                "page": page,
                "score": score
            }
    
    # Display unique sources
    for i, (key, source_info) in enumerate(unique_sources.items(), start=1):
        source = source_info["source"]
        page = source_info["page"]
        score = source_info["score"]
        
        st.write(
            f"**Source {i}:** "
            f"{source} — "
            f"Page {page}  \n"
            f"(relevance: {score:.3f})"
        )


def render_web_sources(web_sources):

    if not web_sources:
        return

    st.markdown("### 🌐 Web Sources")

    for i, result in enumerate(web_sources):

        title = result.get("title", "Untitled")
        url = result.get("url", "")
        score = result.get("score", 0)

        st.markdown(
            f"**Source {i + 1}:** "
            f"[{title}]({url})  \n"
            f"(relevance: {score:.3f})"
        )


# -----------------------------
# PDF Upload
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload PDFs (multiple files supported)",
    type=["pdf"],
    accept_multiple_files=True
)

all_chunks = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        # Check if we already processed this file
        existing_doc = next((doc for doc in st.session_state.documents 
                           if doc["hash"] == file_hash), None)
        
        if existing_doc:
            st.info(f"📄 {uploaded_file.name} already loaded")
            continue
            
        # Process new PDF
        with st.spinner(f"Processing {uploaded_file.name}..."):
            reader = PdfReader(uploaded_file)
            
            pages = []
            
            # Extract text page-by-page
            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):
                page_text = page.extract_text() or ""
                
                pages.append({
                    "text": page_text,
                    "page": page_number,
                    "source": uploaded_file.name
                })
            
            chunks = chunk_pages(pages)
            all_chunks.extend(chunks)
            
            # Store document info
            st.session_state.documents.append({
                "name": uploaded_file.name,
                "hash": file_hash,
                "chunks_count": len(chunks)
            })
            
            st.success(f"📄 {uploaded_file.name} processed: {len(chunks)} chunks")
    
    # Update retriever with all chunks
    if all_chunks:
        st.session_state.retriever = Retriever(all_chunks)
        st.session_state.messages = []  # Clear chat when new docs are added
        
        total_chunks = sum(doc["chunks_count"] for doc in st.session_state.documents)
        st.write(f"**Total:** {total_chunks} chunks from {len(st.session_state.documents)} document(s)")


# -----------------------------
# Display Document Status
# -----------------------------

if st.session_state.documents:
    st.info(f"📚 Loaded {len(st.session_state.documents)} document(s):")
    for doc in st.session_state.documents:
        st.write(f"  • {doc['name']} ({doc['chunks_count']} chunks)")
    
    total_chunks = sum(doc["chunks_count"] for doc in st.session_state.documents)
    st.write(f"**Total chunks:** {total_chunks}")
    
    # Add clear documents button
    if st.button("Clear All Documents"):
        st.session_state.documents = []
        st.session_state.retriever = None
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if (
            message["role"] == "user"
            and "route" in message
        ):

            st.caption(
                f"🧠 Query Route: {message['route']}"
            )

        st.write(message["content"])

        if message["role"] == "assistant":
            render_pdf_sources(message.get("pdf_sources") or [])
            render_web_sources(message.get("web_sources") or [])
# -----------------------------
# Chat Input
# -----------------------------
question = st.chat_input(
    "Ask DeepScout anything..."
)
if question:
    recent_messages = st.session_state.messages[-4:]
    route = classify_query(
        question,
        has_document=(
            st.session_state.retriever is not None
        ),
        recent_messages=recent_messages
    )
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "route": route
    })
    with st.chat_message("user"):
        st.caption(
            f"🧠 Query Route: {route}"
        )
        st.write(question)
    result = run(
        route=route,
        question=question,
        retriever=st.session_state.retriever,
        history=recent_messages
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "pdf_sources": result.pdf_sources,
        "web_sources": result.web_sources,
        "route": result.route
    })
    with st.chat_message("assistant"):

        st.write(result.answer)
        render_pdf_sources(result.pdf_sources)
        render_web_sources(result.web_sources)
