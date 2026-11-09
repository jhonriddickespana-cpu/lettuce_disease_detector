FROM python:3.13-slim

# Install system dependencies required by OpenCV and native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port and run using Gunicorn with Uvicorn worker bound to Railway's $PORT
EXPOSE 8080
CMD sh -c "gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 120"
