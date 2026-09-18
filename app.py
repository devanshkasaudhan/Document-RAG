"""Streamlit entry point for the document Q&A application."""

from __future__ import annotations

import streamlit as st

from config import get_api_key, load_env_file
from generators import generate_with_gemini, generate_with_local, generate_with_openai, get_local_generator
from rag_engine import RAGEngine
from ui import AppSettings, render_sidebar


load_env_file()

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
st.title("Document Q&A")
st.caption("Upload a document, build a searchable index, and get answers grounded in its content.")


@st.cache_resource
def get_engine() -> RAGEngine:
    return RAGEngine()


def process_upload(engine: RAGEngine, upload, ocr_dpi: int) -> None:
    """Extract document text and replace the in-memory search index."""
    file_bytes = upload.getvalue()
    is_pdf = upload.name.lower().endswith(".pdf")

    if not engine.is_embed_loaded():
        with st.status("Loading embedding model…", expanded=True) as status:
            engine.load_embedder()
            status.update(label="Embedding model ready", state="complete")

    text = extract_upload_text(engine, file_bytes, is_pdf, ocr_dpi)
    with st.status("Chunking, embedding & indexing…", expanded=True) as status:
        progress_bar = st.progress(0, text="Embedding…")

        def embed_progress(done: int, total: int) -> None:
            progress_bar.progress(done / total, text=f"Embedding passages: {done}/{total}")

        chunks = engine.build_index(text, progress=embed_progress)
        status.update(label=f"Index built from {chunks} passages — ready for questions.", state="complete")
    st.session_state["ready"] = True


def extract_upload_text(engine: RAGEngine, file_bytes: bytes, is_pdf: bool, ocr_dpi: int) -> str:
    """Extract PDF text or OCR a standalone image, with user-facing progress."""
    if not is_pdf:
        with st.status("Running OCR on image…", expanded=True) as status:
            text = engine.ocr_image(file_bytes)
            if not text.strip():
                status.update(label="No text detected in the image.", state="error")
                st.stop()
            status.update(label=f"Extracted {len(text.split())} words", state="complete")
        return text

    with st.status("Reading document…", expanded=True) as status:
        progress_bar = st.progress(0, text="Starting…")

        def page_status(page: int, total: int, mode: str) -> None:
            label = "reading text" if mode == "text" else "running OCR (image page)"
            progress_bar.progress(page / total, text=f"Page {page}/{total} — {label}")

        result = engine.extract_document(file_bytes, dpi=ocr_dpi, page_status=page_status)
        words = len(result["text"].split())
        if words == 0:
            status.update(label="No readable text found in the document.", state="error")
            st.stop()
        message = f"Extracted {words} words from {result['pages']} pages"
        if result["ocr_pages"]:
            message += f" ({result['ocr_pages']} page(s) via OCR)"
        status.update(label=message, state="complete")
    return result["text"]


def generate_answer(question: str, context: str, settings: AppSettings) -> str:
    """Generate an answer using the backend selected in the sidebar."""
    if settings.backend.startswith("OpenAI"):
        if not settings.api_key:
            st.warning("Add OPENAI_API_KEY to .env or enter it above, or switch to another model.")
            st.stop()
        return generate_with_openai(question, context, settings.api_key)
    if settings.backend.startswith("Gemini"):
        if not settings.api_key:
            st.warning("Add GEMINI_API_KEY to .env or enter it above, or switch to the local model.")
            st.stop()
        return generate_with_gemini(question, context, settings.api_key)

    if "flan_loaded" not in st.session_state:
        with st.status("Loading local answer model (first run)…", expanded=True) as status:
            get_local_generator()
            st.session_state["flan_loaded"] = True
            status.update(label="Local answer model ready", state="complete")
    return generate_with_local(question, context)


def main() -> None:
    engine = get_engine()
    settings = render_sidebar(engine, get_api_key("OpenAI"), get_api_key("Gemini"))

    st.subheader("Step 1 — Upload a document")
    upload = st.file_uploader("PDF, PNG, or JPG", type=["pdf", "png", "jpg", "jpeg"])
    if upload is not None:
        st.success(f"Uploaded: **{upload.name}**  ({len(upload.getvalue()) / 1024:.0f} KB)")

    st.subheader("Step 2 — Process the document")
    if st.button("Extract & build index", type="primary", disabled=upload is None):
        process_upload(engine, upload, settings.ocr_dpi)

    st.subheader("Step 3 — Ask a question")
    question = st.text_input("Your question", placeholder="e.g. Summarize the key findings of this document")
    if st.button("Get answer", disabled=not (question.strip() and st.session_state.get("ready"))):
        with st.spinner("Retrieving relevant passages…"):
            hits = engine.retrieve(question, k=settings.top_k)

        st.markdown("#### Retrieved context")
        for hit in hits:
            with st.expander(f"Passage #{hit['rank']} (distance {hit['distance']:.3f} — lower is closer)"):
                st.write(hit["chunk"])

        st.markdown("#### Answer")
        try:
            with st.spinner("Generating a grounded answer…"):
                context = "\n\n".join(hit["chunk"] for hit in hits)
                answer = generate_answer(question, context, settings)
            st.success(answer)
        except Exception as error:
            st.error(f"Generation failed: {error}")

    st.divider()
    st.caption("Embeddings: all-MiniLM-L6-v2 · Vector search: FAISS · PDF: PyMuPDF · OCR: EasyOCR (image pages only)")


if __name__ == "__main__":
    main()
