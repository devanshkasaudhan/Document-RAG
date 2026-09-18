"""Reusable Streamlit UI components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import streamlit as st


class EmbeddingEngine(Protocol):
    def is_embed_loaded(self) -> bool: ...

    def load_embedder(self): ...


@dataclass(frozen=True)
class AppSettings:
    backend: str
    api_key: str
    top_k: int
    ocr_dpi: int


def render_sidebar(engine: EmbeddingEngine, configured_openai_key: str,
                   configured_gemini_key: str) -> AppSettings:
    """Render sidebar settings and request a key only when one is not configured."""
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
                           expanded=True) as status:
                engine.load_embedder()
                status.update(label="Embedding model loaded", state="complete")
            embed_status.success("Embedding model loaded")

        st.divider()
        backend = st.radio(
            "Answer generator",
            ["Gemini (free API key)", "Local model (free)", "OpenAI (API key)"],
        )
        api_key = _get_selected_key(backend, configured_openai_key, configured_gemini_key)
        top_k = st.slider("Passages to retrieve (k)", 1, 6, 3)
        ocr_dpi = st.select_slider(
            "OCR resolution (DPI)", [150, 200, 300], value=200,
            help="Higher is more accurate but slower. Only affects image pages.",
        )

    return AppSettings(backend=backend, api_key=api_key, top_k=top_k, ocr_dpi=ocr_dpi)


def _get_selected_key(backend: str, openai_key: str, gemini_key: str) -> str:
    if backend.startswith("OpenAI"):
        if openai_key:
            return openai_key
        return st.text_input("OpenAI API key", type="password")
    if backend.startswith("Gemini"):
        if gemini_key:
            return gemini_key
        return st.text_input(
            "Gemini API key", type="password",
            help="Add GEMINI_API_KEY to .env to avoid entering it each time.",
        )
    return ""
