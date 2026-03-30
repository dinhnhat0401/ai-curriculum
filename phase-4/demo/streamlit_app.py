"""
Demo Application Template - Document Q&A with RAG

A Streamlit application that lets users upload documents and ask questions.
Uses RAG (Retrieval-Augmented Generation) to provide grounded answers.

Usage:
    pip install streamlit anthropic
    streamlit run streamlit_app.py

Customize this template for your specific use case.
"""

import os
import time
import streamlit as st

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="",
    layout="wide",
)

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("AI Document Assistant")
    st.markdown("---")
    st.markdown("Upload documents and ask questions. Answers are grounded in your uploaded content.")

    st.markdown("### Settings")
    max_chunks = st.slider("Retrieved chunks", 1, 10, 3)
    show_sources = st.checkbox("Show source chunks", value=True)

    st.markdown("---")
    st.markdown("### Session Stats")
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
        st.session_state.total_latency = 0

    st.metric("Queries", st.session_state.query_count)
    avg_latency = (st.session_state.total_latency / st.session_state.query_count
                   if st.session_state.query_count > 0 else 0)
    st.metric("Avg Latency", f"{avg_latency:.0f}ms")

# ============================================================
# Main Content
# ============================================================

st.title("Document Q&A")
st.markdown("Ask questions about your documents. Powered by RAG.")

# Document upload
uploaded_files = st.file_uploader(
    "Upload documents (.txt files)",
    type=["txt"],
    accept_multiple_files=True,
)

# Process uploaded documents
if uploaded_files:
    with st.expander(f"Uploaded {len(uploaded_files)} document(s)", expanded=False):
        for f in uploaded_files:
            content = f.read().decode("utf-8")
            st.markdown(f"**{f.name}** ({len(content)} characters)")
            st.text(content[:500] + ("..." if len(content) > 500 else ""))
            f.seek(0)  # reset for re-reading

# Chat interface
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            meta = message["metadata"]
            cols = st.columns(3)
            cols[0].caption(f"Latency: {meta.get('latency_ms', 0):.0f}ms")
            cols[1].caption(f"Sources: {meta.get('sources', 'N/A')}")
            cols[2].caption(f"Chunks: {meta.get('chunks_used', 0)}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        start = time.time()

        if not uploaded_files:
            response = "Please upload some documents first, then ask your question."
            sources = "None"
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            response = ("Demo mode: set ANTHROPIC_API_KEY to enable AI responses.\n\n"
                       f"Your question: {prompt}\n"
                       f"Documents loaded: {len(uploaded_files)}")
            sources = ", ".join(f.name for f in uploaded_files)
        else:
            # Build context from uploaded documents
            context_parts = []
            for f in uploaded_files:
                content = f.read().decode("utf-8")
                context_parts.append(f"[Source: {f.name}]\n{content}")
                f.seek(0)

            context = "\n\n---\n\n".join(context_parts)

            # Call Claude
            import anthropic
            client = anthropic.Anthropic()
            api_response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=("Answer questions based ONLY on the provided context. "
                        "If the context doesn't contain the answer, say so. "
                        "Cite which source document you used."),
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {prompt}"
                }],
            )
            response = api_response.content[0].text
            sources = ", ".join(f.name for f in uploaded_files)

        latency = (time.time() - start) * 1000
        st.markdown(response)

        # Show metadata
        metadata = {
            "latency_ms": latency,
            "sources": sources,
            "chunks_used": len(uploaded_files) if uploaded_files else 0,
        }

        if show_sources:
            cols = st.columns(3)
            cols[0].caption(f"Latency: {latency:.0f}ms")
            cols[1].caption(f"Sources: {sources}")
            cols[2].caption(f"Documents: {len(uploaded_files) if uploaded_files else 0}")

        # Update stats
        st.session_state.query_count += 1
        st.session_state.total_latency += latency

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "metadata": metadata,
        })
