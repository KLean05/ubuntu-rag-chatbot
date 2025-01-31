# Dockerfile
FROM python:3.10-slim

# Install system dependencies for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy app code
COPY . .

# Start Ollama and create the model during the build process
RUN ollama serve > /dev/null 2>&1 & \
    sleep 15 && \
    ollama pull mistral

# Expose ports (API + Ollama)
EXPOSE 8000 11434

# Start services
CMD ["sh", "-c", "ollama serve > /dev/null 2>&1 & uvicorn app:app --host 0.0.0.0 --port 8000"]