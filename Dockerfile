# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (important for log streaming)
ENV PYTHONUNBUFFERED=1

# Suppress HuggingFace tokenizer parallelism warning in container
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# ── Install dependencies (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app/      ./app/
COPY retrain/  ./retrain/

# Create logs directory so the file handler doesn't fail on startup
RUN mkdir -p logs

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]