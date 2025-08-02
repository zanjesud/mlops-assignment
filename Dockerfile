# Base stage with common dependencies
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
RUN pip install uv && uv pip install --system .

# Copy common code
COPY models/ ./models/
COPY data/ ./data/
RUN mkdir -p logs

# API stage
FROM base as api
COPY api/ ./api/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# UI stage
FROM base as ui
COPY ui/ ./ui/
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# MLflow stage
FROM base as mlflow
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

# Combined stage (for local development)
FROM base as combined
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*
COPY api/ ./api/
COPY ui/ ./ui/
COPY supervisord.conf ./
EXPOSE 8000 8501 5000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["supervisord", "-c", "/app/supervisord.conf"]