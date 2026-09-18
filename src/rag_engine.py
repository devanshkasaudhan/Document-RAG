import io
import numpy as np
import faiss
import pymupdf  # PyMuPDF
from PIL import Image
from sentence_transformers import SentenceTransformer

TEXT_THRESHOLD = 20


class RAGEngine:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self._embed_model_name = embedding_model
        self.embedder = None
        self._ocr_reader = None
        self.index = None
        self.chunks: list[str] = []


    def is_embed_loaded(self) -> bool:
        return self.embedder is not None

    def load_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(self._embed_model_name)
        return self.embedder

    def load_ocr(self):
        """Load the OCR reader on first use only. Downloads models once."""
        if self._ocr_reader is None:
            import easyocr
            self._ocr_reader = easyocr.Reader(["en"], gpu=False)
        return self._ocr_reader


    def extract_document(self, file_bytes: bytes, dpi: int = 200,
                         page_status=None) -> dict:

        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        total = doc.page_count
        all_text = []
        ocr_pages = 0

        for i, page in enumerate(doc, start=1):
            digital = (page.get_text() or "").strip()

            if len(digital) >= TEXT_THRESHOLD:

                if page_status:
                    page_status(i, total, "text")
                all_text.append(digital)
            else:

                if page_status:
                    page_status(i, total, "ocr")
                reader = self.load_ocr()
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                arr = np.array(img)
                lines = reader.readtext(arr, detail=0, paragraph=True)
                all_text.append(" ".join(lines))
                ocr_pages += 1

        doc.close()
        return {
            "text": "\n".join(all_text),
            "pages": total,
            "ocr_pages": ocr_pages,
        }

    def ocr_image(self, file_bytes: bytes) -> str:
        """OCR a standalone image file (png/jpg)."""
        reader = self.load_ocr()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        arr = np.array(img)
        return " ".join(reader.readtext(arr, detail=0, paragraph=True))


    def chunk_text(self, text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            chunk = " ".join(words[start:start + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks


    def build_index(self, text: str, progress=None) -> int:
        self.load_embedder()
        self.chunks = self.chunk_text(text)
        if not self.chunks:
            raise ValueError("No text could be extracted from this document.")

        vecs, batch, total = [], 64, len(self.chunks)
        for start in range(0, total, batch):
            sub = self.chunks[start:start + batch]
            vecs.append(self.embedder.encode(sub, convert_to_numpy=True,
                                              show_progress_bar=False))
            if progress:
                progress(min(start + batch, total), total)

        embeddings = np.vstack(vecs).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        return total


    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        if self.index is None:
            raise ValueError("Index not built yet. Process a document first.")
        qv = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(qv, k)
        return [
            {"rank": r, "chunk": self.chunks[i], "distance": float(d)}
            for r, (i, d) in enumerate(zip(indices[0], distances[0]), start=1)
        ]
