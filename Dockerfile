# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and supervisor
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN pip install uv && uv pip install --system .

# Copy application code
COPY api/ ./api/
COPY ui/ ./ui/
COPY models/ ./models/
COPY data/ ./data/
COPY supervisord.conf ./

RUN mkdir -p logs


# Expose ports for API and UI
EXPOSE 8000 8501

# Healthcheck (optional, checks API)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["supervisord", "-c", "/app/supervisord.conf"]