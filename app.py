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

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    needs_reprocess = (
        st.session_state.document_name != uploaded_file.name
        or st.session_state.document_hash != file_hash
    )

    if needs_reprocess:

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

        st.session_state.retriever = Retriever(chunks)
        st.session_state.document_name = uploaded_file.name
        st.session_state.document_hash = file_hash
        st.session_state.messages = []

        st.success(
            f"📄 {uploaded_file.name} processed successfully!"
        )

        st.write(
            f"Created {len(chunks)} chunks."
        )


# -----------------------------
# Display Document Status
# -----------------------------

if st.session_state.retriever is not None:

    st.info(
        f"📚 Document ready: "
        f"{st.session_state.document_name}"
    )


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
