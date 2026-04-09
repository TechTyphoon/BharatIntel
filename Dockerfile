FROM python:3.12-slim

WORKDIR /app

# Install system deps: build tools + WeasyPrint rendering libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than full CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create output directory
RUN mkdir -p output/sample logs

# Expose port
EXPOSE 8080

# Start server — Render injects PORT env var (default 8080)
CMD uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8080}
