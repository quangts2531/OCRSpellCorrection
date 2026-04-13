# ============================================================
# Dockerfile — Hugging Face Spaces (Docker SDK)
# Custom OCR pipeline. No Tesseract.
# ============================================================

FROM python:3.9-slim

# ----- System dependencies for OpenCV & image processing -----
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*



# ----- Create non-root user (HF Spaces requirement, UID 1000) -----
RUN useradd -m -u 1000 appuser

# ----- Set working directory -----
WORKDIR /app

# ----- Install Python deps (cached layer) -----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----- Copy application source -----
COPY . .

# ----- Ensure appuser owns everything -----
RUN chown -R appuser:appuser /app

# ----- HuggingFace model cache -----
ENV HF_HOME=/app/.cache/huggingface

# ----- Switch to non-root user -----
USER appuser

# ----- Expose HF Spaces required port -----
EXPOSE 7860

# ----- Run with Gunicorn -----
CMD ["gunicorn", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "2", \
     "--timeout", "120", \
     "app:app"]
