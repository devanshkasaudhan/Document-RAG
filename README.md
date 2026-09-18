# Document Q&A RAG Application

## Overview

This project is a local, browser-based document question-answering application. It lets a user upload a PDF or image, extracts the document text, creates a searchable vector index, retrieves the passages most relevant to a question, and uses a selected language model to produce an answer grounded in those passages.

The application is designed to make document exploration practical without sending the original document to an answer provider by default. Document processing, OCR, embeddings, and vector search run locally. Only the retrieved text passages and the user's question are sent to OpenAI or Gemini when one of those hosted answer generators is selected. A fully local answer-generator option is also available.

The web interface is built with Streamlit and runs on port 8501.

## What the application supports

- PDF uploads, including text-based PDFs and scanned or image-only pages.
- PNG, JPG, and JPEG uploads through local OCR.
- Direct text extraction from PDFs using PyMuPDF.
- Automatic English OCR fallback using EasyOCR when a PDF page has too little extractable text.
- Semantic search over document passages using the `all-MiniLM-L6-v2` sentence-transformer model.
- In-memory nearest-neighbor retrieval with FAISS.
- Grounded answer generation with Gemini, OpenAI, or a local FLAN-T5 model.
- A configurable number of retrieved passages and OCR resolution.

## How it works

The project follows a small retrieval-augmented generation (RAG) pipeline:

```text
Upload document
      |
      v
Extract native PDF text or run OCR
      |
      v
Split text into overlapping passages
      |
      v
Create embeddings and build a FAISS index
      |
      v
Embed the user question and retrieve nearest passages
      |
      v
Send the question plus retrieved context to the selected generator
      |
      v
Display the answer and the retrieved source passages
```

### 1. Document extraction

For PDFs, `RAGEngine.extract_document` checks each page for native text. Pages with at least 20 extracted characters are treated as digital text. Pages below that threshold are rendered to an image at the configured DPI and processed by EasyOCR. Standalone image uploads always use EasyOCR.

This hybrid approach avoids unnecessary OCR work for normal digital PDFs while allowing scanned paperwork and photographed content to remain searchable.

![Document extraction](images\Screenshot_18-9-2026_205549_localhost.jpeg)


### 2. Chunking and embedding

After extraction, the document is split by words into passages of 120 words with a 20-word overlap. The overlap helps preserve context at the boundary between passages. The application embeds each passage with `all-MiniLM-L6-v2`, then stores the float32 vectors in a `faiss.IndexFlatL2` index.

The index and its source passages live only in application memory. Processing a new upload replaces the active index. There is no database or long-term document storage in the application as it currently stands.



### 3. Retrieval

When the user asks a question, the same embedding model converts the question into a vector. FAISS returns the nearest matching passages using L2 distance. The user can choose between one and six passages in the sidebar; the default is three.

The interface shows every retrieved passage and its distance before showing the generated answer. Lower distance values mean that the embedding model considers a passage more similar to the question. This visibility makes it easier to judge whether the answer has useful supporting context.

![Alt text](images\Screenshot_18-9-2026_204936_localhost.jpeg)

### 4. Answer generation

The application builds a prompt from the question and retrieved passages. The hosted generator prompts instruct the model to use only that context and to say it does not know when the answer is absent.

Three answer-generation options are available:

| Option | Model | API key | Notes |
| --- | --- | --- | --- |
| Gemini | `gemini-2.5-flash` | `GEMINI_API_KEY` | Hosted option requiring a Gemini key. |
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` | Hosted option requiring an OpenAI key. |
| Local model | `google/flan-t5-base` | Not required | Downloaded on first use; may be slower and less capable than hosted models. |

![Answer Generation](images\Screenshot_18-9-2026_204947_localhost.jpeg)


## Project structure

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit entry point and the upload, indexing, retrieval, and answer workflow. |
| `ui.py` | Sidebar controls and application settings. |
| `rag_engine.py` | Text extraction, OCR, chunking, embeddings, FAISS indexing, and retrieval. |
| `generators.py` | OpenAI, Gemini, and local FLAN-T5 answer generator integrations. |
| `config.py` | Lightweight `.env` loading and API-key lookup. |
| `requirements.txt` | Python dependencies. |
| `Dockerfile` | Production-style Streamlit image with CPU PyTorch and native OCR libraries. |
| `docker-compose.yml` | Local container service, port mapping, API-key forwarding, and persistent model cache. |

## Running locally

### Prerequisites

- Python 3.12 is recommended.
- Internet access is needed on first use to download Python packages and model files.
- An OpenAI or Gemini API key is optional; it is only needed when using that provider.

### Install and launch

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` in a browser after Streamlit starts.

### Configure hosted providers

Create a local `.env` file in the project root when using a hosted generator:

```dotenv
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

Keys are loaded locally by `config.py`; existing environment variables take precedence. Do not commit `.env` files. The provided Docker configuration forwards these variables into the container without embedding them in the image.

## Running with Docker

Docker packages the application and its system dependencies, including the OpenGL and GLib libraries required by EasyOCR. The Compose service is named `rag-app` and maps the container's Streamlit port to the host.

```powershell
docker compose up --build --detach
```
![Docker Container running](images\image.png)


Then open `http://localhost:8501`.

Useful service commands:

```powershell
docker compose ps
docker compose logs --follow rag-app
docker compose down
```

The named `hf-cache` volume persists Hugging Face and EasyOCR model downloads between container restarts. The service runs as the non-root `appuser` account and includes a health check at `/_stcore/health`.

![Hugging Face Volume](images\hf_volume.png)


## Using the interface

1. Choose an answer generator in the sidebar. Add an API key only if the selected hosted provider needs one.
2. Upload a PDF, PNG, JPG, or JPEG file.
3. Select **Extract & build index**. First use may take longer while the embedding and OCR models download.
4. Enter a question and select **Get answer**.
5. Review the retrieved passages along with the answer. If the passages do not contain the needed information, rephrase the question, increase the number of retrieved passages, or verify the uploaded document.

## Operational notes and limitations

- The index is in memory, so it is lost when the application process restarts and is replaced by the next processed document.
- OCR is configured for English and can make errors with poor scans, handwriting, unusual fonts, or non-English content.
- Retrieval quality depends on the extracted text, the fixed chunk size, the embedding model, and the selected number of passages.
- Hosted generators receive the question and retrieved context, not necessarily the full document. They still require credentials and network access.
- The local answer model and OCR/embedding models are downloaded on first use, which can take time and disk space.
- The generated answer should be treated as an aid to document review. Check it against the displayed retrieved passages, especially for important decisions or sensitive information.

## Extending the project

Useful next improvements include persistent document/index storage, source-page citations, multi-document collections, metadata-aware filtering, better chunking for headings and sentences, configurable embedding models, automated tests, and provider-specific retry/error handling.
