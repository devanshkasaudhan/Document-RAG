FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# EasyOCR needs these shared libraries to process images in the container.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install matching CPU builds together.  Installing torch alone lets a later
# dependency pull an incompatible torchvision wheel, which breaks the
# torchvision::nms registration during transformers import.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.7.1+cpu torchvision==0.22.1+cpu \
    && python -m pip install -r requirements.txt

COPY src/ ./src/

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir --parents "$HF_HOME" \
    && chown --recursive appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
