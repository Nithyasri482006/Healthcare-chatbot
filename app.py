import streamlit as st

from utils.pdf_reader import extract_pdf_text
from utils.text_splitter import split_text

from rag.embeddings import create_embeddings
from rag.vector_store import create_vector_store
from rag.retriever import retrieve_context

from hf_chat import ask_ai

st.set_page_config(
    page_title="Healthcare AI Chatbot",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare AI Chatbot")

uploaded_file = st.file_uploader(
    "Upload Healthcare PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("✅ File Uploaded Successfully!")

    try:
        # -----------------------------
        # Extract Text
        # -----------------------------
        text = extract_pdf_text(uploaded_file)

        if not text.strip():
            st.error("❌ No text could be extracted from this PDF.")
            st.info("The PDF may be scanned or image-based.")
            st.stop()

        st.write("### Extracted Text Length")
        st.write(len(text))

        with st.expander("View Extracted Text"):
            st.text(text[:5000])

        # -----------------------------
        # Split into Chunks
        # -----------------------------
        chunks = split_text(text)

        st.success(f"Number of chunks: {len(chunks)}")

        with st.expander("View Chunks"):
            for i, chunk in enumerate(chunks):
                st.write(f"### Chunk {i+1}")
                st.write(chunk)

        # -----------------------------
        # Create Embeddings
        # -----------------------------
        with st.spinner("Creating embeddings..."):
            embeddings = create_embeddings(chunks)

        # -----------------------------
        # Create Vector Store
        # -----------------------------
        vector_store = create_vector_store(embeddings)

        st.divider()

        st.subheader("💬 Ask a Question")

        question = st.text_input(
            "Enter your question"
        )

        if question:

            with st.spinner("Searching relevant information..."):

                context = retrieve_context(
                    question,
                    vector_store,
                    chunks
                )

            st.subheader("Retrieved Context")

            if context:
                st.write(context)
            else:
                st.warning("No relevant context found.")

            with st.spinner("Generating Answer..."):

                try:
                    answer = ask_ai(
                        context,
                        question
                    )

                    st.subheader("Answer")
                    st.success(answer)

                except Exception as e:
                    st.error("AI Response Error")
                    st.exception(e)

    except Exception as e:
        st.error("Application Error")
        st.exception(e)