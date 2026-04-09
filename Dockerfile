FROM python:3.12-slim

WORKDIR /app

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
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
