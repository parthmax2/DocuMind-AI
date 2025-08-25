# ==========================
# Base image
# ==========================
FROM python:3.11-slim

# ==========================
# System dependencies
# ==========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ghostscript \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==========================
# Set working directory
# ==========================
WORKDIR /app

# ==========================
# Install Python dependencies
# ==========================
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ==========================
# Copy app code
# ==========================
COPY . .

# ==========================
# Hugging Face cache setup
# ==========================
# Use /tmp/hf_cache because it's always writable on Hugging Face Spaces
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_DATASETS_CACHE=/tmp/hf_cache

RUN mkdir -p /app/uploads /app/summaries /app/embeddings /app/logs /tmp/hf_cache \
    && chmod -R 777 /app /tmp/hf_cache

# ==========================
# (Optional) Pre-download SentenceTransformer model
# Speeds up startup by caching during build
# ==========================
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ==========================
# Expose port
# ==========================
EXPOSE 7860

# ==========================
# Command to run FastAPI app
# ==========================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
