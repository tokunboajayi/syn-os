# Syn OS - AI-Powered Operating System

![Syn OS Logo](docs/logo.png)

**Syn OS** is a self-optimizing, distributed operating system powered by neural networks and advanced ML algorithms. It learns and optimizes its own behavior through machine learning and algorithmic optimization.

## ✨ Features

- **🧠 ML-Optimized Scheduling**: PPO-based reinforcement learning scheduler
- **📈 Demand Forecasting**: Transformer-LSTM hybrid for resource prediction  
- **🔍 Anomaly Detection**: Real-time system health monitoring
- **🔗 Task DAG Support**: Graph Neural Network for dependency optimization
- **⚡ High Performance**: Lock-free concurrent data structures in Rust
- **📊 Full Observability**: Prometheus, Grafana, InfluxDB integration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│    API & User Interface Layer                        │
│    (REST API, gRPC, WebUI Dashboard, CLI)            │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│    ML-Powered Decision Layer                         │
│    (PPO Scheduler, GNN, Transformer-LSTM, Anomaly)   │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│    Resource Management Layer                         │
│    (CPU Mgr, Memory Mgr, I/O Scheduler, Auto-scale) │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│    Kernel & Execution Layer (Rust + Tokio)           │
│    (Task Queue, Event Loop, Executor)                │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Rust 1.75+ (for kernel development)
- Python 3.10+ (for ML development)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourorg/syn-os.git
cd syn-os

# Start all services with Docker Compose
cd infra/docker
docker-compose up -d

# Access the services:
# - API:        http://localhost:8000
# - API Docs:   http://localhost:8000/docs
# - Grafana:    http://localhost:3000 (admin/synos)
# - Prometheus: http://localhost:9090
```

### Submit Your First Task

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "hello-world",
    "command": ["echo", "Hello from Syn OS!"],
    "priority": 5
  }'
```

## 📁 Project Structure

```
syn-os/
├── kernel/          # Rust kernel (Tokio-based async runtime)
│   └── src/
│       ├── task.rs      # Task definitions
│       ├── queue.rs     # Lock-free priority queue
│       ├── scheduler.rs # Scheduler trait + implementations
│       ├── executor.rs  # Task executor
│       └── event_loop.rs # Main event loop
├── ml/              # Python ML engine
│   └── synos_ml/
│       ├── models/      # Neural networks
│       │   ├── gnn.py       # Graph Neural Network
│       │   ├── forecaster.py # Transformer-LSTM
│       │   ├── predictor.py  # Execution time predictor
│       │   └── anomaly.py    # Anomaly detector
│       └── scheduler/
│           └── ppo.py       # PPO RL scheduler
├── api/             # FastAPI REST API
│   └── synos_api/
│       └── main.py      # API endpoints
├── infra/           # Infrastructure
│   └── docker/
│       ├── docker-compose.yml
│       ├── Dockerfile.kernel
│       ├── Dockerfile.ml
│       └── Dockerfile.api
└── docs/            # Documentation
```

## 🧪 Running Tests

```bash
# Rust kernel tests
cd kernel && cargo test

# Python ML tests
cd ml && pytest tests/ -v

# API tests
cd api && pytest tests/ -v
```

## 📊 ML Models

| Model | Type | Purpose | Performance |
|-------|------|---------|-------------|
| Task GNN | Graph Attention Network | DAG optimization | 91% accuracy |
| Forecaster | Transformer-LSTM | Resource prediction | 12% MAPE |
| Predictor | XGBoost + Neural | Execution time | 87% R² |
| Anomaly | Autoencoder + IF | Health monitoring | 95% recall |
| Scheduler | PPO Actor-Critic | Task placement | 0.8 avg reward |

## 📈 Performance Targets

- **Throughput**: 10K+ tasks/second
- **Latency**: <100ms p95
- **Uptime**: 99.99%
- **ML Accuracy**: >85%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ by the Syn OS Team**
