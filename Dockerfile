FROM python:3.11-slim

# System deps for TensorFlow + PostgreSQL (psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 pkg-config \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create model cache dir
# On Render: mount a Disk at /data, set MODEL_CACHE_DIR=/data/model_cache
# On free tier (no disk): defaults to /tmp/model_cache (wiped on redeploy)
RUN mkdir -p /tmp/model_cache

EXPOSE 8080

# Single worker + threads avoids duplicate scheduler runs
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", \
     "--threads", "4", "--timeout", "300", "--keep-alive", "5", "app:app"]
