# Dockerfile for the FastAPI backend.
# Build:  docker build -t knowrag-backend ./backend

# Start from the official slim Python image (Debian-based, small, has pip).
FROM python:3.12-slim

# Don't write .pyc files, and print logs immediately instead of buffering them.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# All following commands run inside /app in the container.
WORKDIR /app

# Copy ONLY the requirements first: Docker caches this layer, so dependencies are
# re-installed only when requirements.txt changes, not on every code edit.
COPY requirements.txt .

# Install the Python dependencies without keeping pip's download cache (smaller image).
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code.
COPY app ./app

# Store the database and downloaded embedding model in /data (mounted as a volume).
ENV KNOWRAG_DATA_DIR=/data

# Document that the server listens on port 8000.
EXPOSE 8000

# Start Uvicorn; 0.0.0.0 makes it reachable from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
