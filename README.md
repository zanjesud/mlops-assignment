# MLOps Assignment - Dockerized Application

This project demonstrates a complete MLOps pipeline with FastAPI REST API, Streamlit UI, MLflow model tracking, Prometheus monitoring, and Grafana visualization.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │   MLflow Server │
│   (Port: 8501)  │◄──►│   (Port: 8000)  │◄──►│   (Port: 5000)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Prometheus    │
                    │   (Port: 9090)  │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Grafana     │
                    │   (Port: 3000)  │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Git

### Step 1: Clone and Setup

```bash
git clone <your-repo-url>
cd mlops-assignment
```

### Step 2: Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### Step 3: Access Services

- **Streamlit UI**: http://localhost:8501
- **FastAPI API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 📋 Step-by-Step Dockerization Guide

### 1. Application Structure Analysis

Your application consists of:
- **FastAPI Backend** (`api/main.py`): REST API with Prometheus metrics
- **Streamlit Frontend** (`ui/app.py`): User interface
- **MLflow Integration**: Model tracking and serving
- **DVC**: Data version control

### 2. Dockerfile Creation

#### For FastAPI (Dockerfile.api):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv pip install --system .
COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### For Streamlit (Dockerfile.ui):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv pip install --system .
COPY ui/ ./ui/
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### 3. Docker Compose Orchestration

The `docker-compose.yml` file orchestrates:
- **MLflow Server**: Model tracking and serving
- **FastAPI Application**: REST API with metrics
- **Streamlit UI**: User interface
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Redis**: Caching (optional)

### 4. Monitoring Setup

#### Prometheus Configuration (`monitoring/prometheus.yml`):
```yaml
scrape_configs:
  - job_name: 'fastapi-app'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

#### Grafana Setup:
- Datasource: Prometheus
- Dashboard: MLOps monitoring dashboard
- Metrics: API request rate, response time, error rate

### 5. CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) includes:

1. **Data Pipeline**: Data processing and DVC push
2. **Model Training**: Automated model training with MLflow
3. **Model Evaluation**: Performance threshold checking
4. **Build & Test**: Docker image building and testing
5. **Deployment**: Staging and production deployments
6. **Security Scan**: Vulnerability scanning with Trivy

## 🔧 Development Workflow

### Local Development

```bash
# Install dependencies
pip install uv
uv pip install --system .

# Run MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Run FastAPI (in another terminal)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit (in another terminal)
streamlit run ui/app.py
```

### Model Training

```bash
# Train model
python models/train.py --model_type rf

# Check performance
python scripts/check_model_performance.py
```

### Testing

```bash
# Run integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Run unit tests
pytest tests/
```

## 📊 Monitoring and Observability

### Prometheus Metrics

The FastAPI application exposes metrics at `/metrics`:
- HTTP request rate
- Response time percentiles
- Error rates
- Custom business metrics

### Grafana Dashboards

Pre-configured dashboards include:
- API performance metrics
- Model prediction statistics
- System health indicators
- Error rate monitoring

### Health Checks

- **API Health**: `GET /health`
- **Docker Health**: Built-in health checks in Dockerfiles
- **Service Dependencies**: Proper service orchestration

## 🚀 Deployment

### Production Deployment

1. **Build Images**:
```bash
docker build -f Dockerfile.api -t your-registry/api:latest .
docker build -f Dockerfile.ui -t your-registry/ui:latest .
```

2. **Deploy with Docker Compose**:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

3. **Kubernetes Deployment** (optional):
```bash
kubectl apply -f k8s/
```

### Environment Variables

```bash
# API Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
API_URL=http://api:8000

# Grafana Configuration
GF_SECURITY_ADMIN_PASSWORD=admin
```

## 🔒 Security Considerations

1. **Image Scanning**: Trivy vulnerability scanning in CI/CD
2. **Secrets Management**: Use Docker secrets or Kubernetes secrets
3. **Network Security**: Isolated Docker networks
4. **Access Control**: Grafana authentication

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale API instances
docker-compose up --scale api=3

# Load balancing with nginx
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up
```

### Performance Optimization

1. **Multi-stage Docker builds**
2. **Layer caching optimization**
3. **Resource limits and requests**
4. **Connection pooling**

## 🐛 Troubleshooting

### Common Issues

1. **Port Conflicts**: Check if ports 8000, 8501, 5000, 9090, 3000 are available
2. **MLflow Connection**: Ensure MLflow server is running before API
3. **Model Loading**: Verify model is registered in MLflow
4. **Prometheus Scraping**: Check network connectivity between services

### Debug Commands

```bash
# Check service logs
docker-compose logs api
docker-compose logs ui

# Check service health
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Docker Documentation](https://docs.docker.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
