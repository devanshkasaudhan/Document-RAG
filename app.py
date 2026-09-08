import os
import streamlit as st
from rag_engine import RAGEngine
from generators import generate_with_openai, generate_with_gemini, generate_with_local, get_local_generator

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
st.title("Document Q&A")
st.caption("Upload a document, build a searchable index, and get answers grounded in its content.")


@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    st.subheader("Model status")
    embed_status = st.empty()
    if engine.is_embed_loaded():
        embed_status.success("Embedding model loaded")
    else:
        embed_status.info("Embedding model not loaded yet")

    if st.button("Preload embedding model"):
        with st.status("Loading embedding model (first run downloads ~90 MB)…",
                       expanded=True) as s:
            engine.load_embedder()
            s.update(label="Embedding model loaded", state="complete")
        embed_status.success("Embedding model loaded")

    st.divider()
    backend = st.radio("Answer generator",
                       ["Gemini (free API key)", "Local model (free)", "OpenAI (API key)"])
    api_key = ""
    gemini_key = ""
    if backend.startswith("OpenAI"):
        api_key = st.text_input("OpenAI API key", type="password",
                                value=os.getenv("OPENAI_API_KEY", ""))
    elif backend.startswith("Gemini"):
        gemini_key = st.text_input("Gemini API key", type="password",
                                   value=os.getenv("GEMINI_API_KEY", ""),
                                   help="Get a free key at aistudio.google.com/apikey")
    top_k = st.slider("Passages to retrieve (k)", 1, 6, 3)
    ocr_dpi = st.select_slider("OCR resolution (DPI)", [150, 200, 300], value=200,
                               help="Higher is more accurate but slower. Only affects image pages.")


st.subheader("Step 1 — Upload a document")
upload = st.file_uploader("PDF, PNG, or JPG", type=["pdf", "png", "jpg", "jpeg"])
if upload is not None:
    st.success(f"Uploaded: **{upload.name}**  ({len(upload.getvalue()) / 1024:.0f} KB)")


st.subheader("Step 2 — Process the document")

if st.button("Extract & build index", type="primary", disabled=upload is None):
    file_bytes = upload.getvalue()
    is_pdf = upload.name.lower().endswith(".pdf")


    if not engine.is_embed_loaded():
        with st.status("Loading embedding model…", expanded=True) as s:
            engine.load_embedder()
            s.update(label="Embedding model ready", state="complete")


    if is_pdf:
        with st.status("Reading document…", expanded=True) as s:
            bar = st.progress(0, text="Starting…")

            def page_status(page, total, mode):
                label = "reading text" if mode == "text" else "running OCR (image page)"
                bar.progress(page / total, text=f"Page {page}/{total} — {label}")

            result = engine.extract_document(file_bytes, dpi=ocr_dpi,
                                             page_status=page_status)
            words = len(result["text"].split())
            if words == 0:
                s.update(label="No readable text found in the document.",
                         state="error")
                st.stop()
            msg = f"Extracted {words} words from {result['pages']} pages"
            if result["ocr_pages"]:
                msg += f" ({result['ocr_pages']} page(s) via OCR)"
            s.update(label=msg, state="complete")
        text = result["text"]
    else:
        # Standalone image -> OCR directly
        with st.status("Running OCR on image…", expanded=True) as s:
            text = engine.ocr_image(file_bytes)
            if not text.strip():
                s.update(label="No text detected in the image.", state="error")
                st.stop()
            s.update(label=f"Extracted {len(text.split())} words", state="complete")


    with st.status("Chunking, embedding & indexing…", expanded=True) as s:
        ebar = st.progress(0, text="Embedding…")

        def embed_progress(done, total):
            ebar.progress(done / total, text=f"Embedding passages: {done}/{total}")

        n = engine.build_index(text, progress=embed_progress)
        s.update(label=f"Index built from {n} passages — ready for questions.",
                 state="complete")

    st.session_state["ready"] = True


st.subheader("Step 3 — Ask a question")
question = st.text_input("Your question",
                         placeholder="e.g. Summarize the key findings of this document")

if st.button("Get answer",
             disabled=not (question.strip() and st.session_state.get("ready"))):
    with st.spinner("Retrieving relevant passages…"):
        hits = engine.retrieve(question, k=top_k)

    st.markdown("#### Retrieved context")
    for h in hits:
        with st.expander(f"Passage #{h['rank']} (distance {h['distance']:.3f} — lower is closer)"):
            st.write(h["chunk"])

    context = "\n\n".join(h["chunk"] for h in hits)

    st.markdown("#### Answer")
    try:
        with st.spinner("Generating a grounded answer…"):
            if backend.startswith("OpenAI"):
                if not api_key:
                    st.warning("Add your OpenAI key in the sidebar, or switch to the free local model.")
                    st.stop()
                answer = generate_with_openai(question, context, api_key)
            elif backend.startswith("Gemini"):
                if not gemini_key:
                    st.warning("Add your free Gemini key in the sidebar (aistudio.google.com/apikey), "
                               "or switch to the local model.")
                    st.stop()
                answer = generate_with_gemini(question, context, gemini_key)
            else:
                if "flan_loaded" not in st.session_state:
                    with st.status("Loading local answer model (first run)…",
                                   expanded=True) as s:
                        get_local_generator()
                        st.session_state["flan_loaded"] = True
                        s.update(label="Local answer model ready", state="complete")
                answer = generate_with_local(question, context)
        st.success(answer)
    except Exception as e:
        st.error(f"Generation failed: {e}")

st.divider()
st.caption("Embeddings: all-MiniLM-L6-v2 · Vector search: FAISS · PDF: PyMuPDF · OCR: EasyOCR (image pages only)")