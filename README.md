# Syn OS — AI-Powered Neural Operating System

> **Self-optimizing, distributed OS backend powered by Rust + Python ML**  
> Real-time threat detection · GIS tracking · Voice control · Device management

---

## ✨ Features

### 🧠 AI / ML Engine
| Model | Type | Purpose | Performance |
|-------|------|---------|-------------|
| Task GNN | Graph Attention Network | DAG dependency optimization | 91% accuracy |
| Forecaster | Transformer-LSTM | Resource demand prediction | 12% MAPE |
| Predictor | XGBoost + Neural ensemble | Execution time estimation | 87% R² |
| Anomaly Detector | Autoencoder + Isolation Forest | Health & intrusion detection | 95% recall |
| PPO Scheduler | Actor-Critic RL | ML-optimized task placement | 0.8 avg reward |
| Synapse VAE | Variational Autoencoder | Hardware fingerprinting + self-improvement | — |
| NIDS | PyTorch neural net | Network intrusion detection | — |
| VulnPredictor | Gradient boosted classifier | Port/vuln prediction | — |
| TrajectoryPredictor | LSTM | GPS path prediction | — |

### 🔒 Security & Penetration Testing
- **Network Scanner** — async Rust port scanner with nmap/masscan integration
- **AI-Powered IDS** — real-time intrusion detection via ML anomaly scoring
- **Vulnerability Predictor** — predicts open ports and likely CVEs before scanning
- **Threat Intelligence Feed** — live IOC feed (AlienVault OTX), IP/domain checker, severity filter
- **Rate limiting** — 100 req/min per client; circuit breaker for external calls

### 🗺️ Geospatial & Location Tracking
- **Outdoor Map** — Leaflet dark-mode map, live device positions, predicted paths
- **Indoor Mode** — upload floor plans, overlay live device locations indoors
- **Trajectory Predictor** — LSTM model forecasting device paths 5 min ahead
- **Self-hosted TileServer** — fully offline, private map tile serving

### 📡 Device Management *(NEW)*
- Register/update/delete tracked network devices
- Per-device live ping → online/offline/unknown status
- Pre-seeded demo devices for instant demo

### ⚠ Threat Intelligence *(NEW)*
- `/api/v1/threat-intel/feed` — paginated IOC feed with 5-min cache
- `/api/v1/threat-intel/check` — check any IP or domain against the feed
- Offline fallback dataset — always works without external API

### 🤖 Synapse Self-Improvement Core
- Nightly self-training pipeline (`train_online.py`)
- Experience replay buffer with disk persistence
- `/api/v1/synapse` health-score, ingest, train, and replay-stats routes

### ⚡ Rust Kernel
- Async Tokio runtime with lock-free priority queue
- AI-driven fan/clock hardware control
- Wine/Proton Windows app compatibility layer
- ArchISO integration for bare-metal deployment

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  syn-os-edex  (Svelte/TS frontend — eDEX-UI shell)           │
│  Tabs: Tasks · Security · Threat Intel · Geo · Devices · ML  │
│  Voice: "Synapse, <command>" → Web Speech API navigation      │
└─────────────────────────┬────────────────────────────────────┘
                          │ REST + WebSocket
┌─────────────────────────▼────────────────────────────────────┐
│  FastAPI Gateway  (syn-os/api)                                │
│  /tasks · /security · /scanner · /geo · /synapse             │
│  /threat-intel · /devices  ← NEW                             │
└──────┬───────────────────┬──────────────────┬────────────────┘
       │                   │                  │
┌──────▼──────┐  ┌─────────▼────────┐  ┌─────▼──────────────┐
│ Rust Kernel │  │  Python ML Engine │  │  Infrastructure     │
│ (Tokio)     │  │  (PyTorch/XGBoost)│  │  Docker · k8s      │
│ port scan   │  │  anomaly, nlp,    │  │  Prometheus/Grafana │
│ async queue │  │  trajectory, NIDS │  │  Redis · PostGIS    │
└─────────────┘  └──────────────────┘  └────────────────────┘
```

---

## 📁 Project Structure

```
syn-os/
├── kernel/                  # Rust async kernel (Tokio)
│   └── src/
│       ├── main.rs
│       ├── task.rs          # Task types
│       ├── queue.rs         # Lock-free priority queue
│       ├── scheduler.rs     # Scheduler trait
│       ├── executor.rs      # Task executor
│       ├── scanner.rs       # Async network port scanner
│       └── hardware.rs      # AI fan/clock control
│
├── ml/                      # Python ML engine
│   └── synos_ml/
│       ├── models/
│       │   ├── gnn.py           # Graph Attention Net (task DAG)
│       │   ├── forecaster.py    # Transformer-LSTM (demand)
│       │   ├── predictor.py     # XGBoost execution predictor
│       │   ├── anomaly.py       # Autoencoder anomaly detector
│       │   ├── fingerprinter.py # Synapse VAE
│       │   ├── nids.py          # Network IDS neural net
│       │   ├── vuln_predictor.py# Vulnerability predictor
│       │   └── trajectory.py    # LSTM path predictor
│       ├── core/
│       │   ├── replay_buffer.py # Experience replay (disk-backed)
│       │   └── scheduler.py     # PPO actor-critic
│       ├── serving/
│       │   └── server.py        # ML serving endpoint
│       └── training/
│           └── trainer.py
│
├── api/                     # FastAPI REST API
│   └── synos_api/
│       ├── main.py          # App factory + router registration
│       ├── core/
│       │   ├── scheduler.py
│       │   ├── socket.py    # Socket.IO server
│       │   └── mock_kernel.py
│       └── routes/
│           ├── security.py      # Security monitoring
│           ├── scanner.py       # Network scan endpoints
│           ├── geo.py           # GIS / location endpoints
│           ├── synapse.py       # Self-improvement endpoints
│           ├── threat_intel.py  # IOC feed + IP check  ← NEW
│           └── devices.py       # Device CRUD + ping   ← NEW
│
├── infra/
│   ├── docker/
│   │   ├── docker-compose.yml
│   │   ├── Dockerfile.kernel
│   │   ├── Dockerfile.ml
│   │   └── Dockerfile.api
│   ├── k8s/                 # Kubernetes manifests
│   └── kiosk/               # Bare-metal kiosk scripts
│
├── scripts/
│   ├── train_security_models.py
│   ├── train_trajectory_model.py
│   ├── verify_scanner.py
│   └── war_games_simulation.py
│
└── tests/
    ├── integration/
    ├── ml/
    └── verify_system.py
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Rust 1.75+ (kernel dev)
- Python 3.10+ (ML dev)
- Node 18+ (frontend dev — see syn-os-edex)

### Start All Services
```bash
git clone https://github.com/tokunboajayi/syn-os.git
cd syn-os/infra/docker
docker-compose up -d
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

### Key API Examples
```bash
# Submit a task
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"name":"hello","command":["echo","hi"],"priority":5}'

# Get threat intel feed
curl http://localhost:8000/api/v1/threat-intel/feed

# Check an IP
curl -X POST http://localhost:8000/api/v1/threat-intel/check \
  -H "Content-Type: application/json" \
  -d '{"value":"185.220.101.45"}'

# List devices
curl http://localhost:8000/api/v1/devices

# Register a device
curl -X POST http://localhost:8000/api/v1/devices \
  -H "Content-Type: application/json" \
  -d '{"name":"Lab Workstation","ip":"192.168.1.50","type":"workstation"}'

# Ping a device
curl -X POST http://localhost:8000/api/v1/devices/{id}/ping
```

---

## 🧪 Tests
```bash
# Rust kernel
cd kernel && cargo test

# Python ML
cd ml && pytest tests/ -v

# API
cd api && pytest tests/ -v

# Security simulation
python scripts/war_games_simulation.py
```

---

## 🗺️ Roadmap — Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Security Core (Rust scanner, async port scan, Docker tooling) | ✅ |
| 2 | AI Pen-Testing (NIDS, VulnPredictor, ML→scan pipeline) | ✅ |
| 3 | GIS & Tracking (PostGIS, TileServer, Indoor Mode, LSTM paths) | ✅ |
| 4 | Frontend (syn-os-edex War Games HUD, WebSocket real-time) | ✅ |
| 5 | Optimization (rate limiting, circuit breaker, verification) | ✅ |
| 6 | Bare Metal (ArchISO, Plymouth boot, kiosk shell, Wine/Proton) | ✅ |
| 7 | Synapse AI Core (VAE fingerprinter, experience replay, nightly train) | ✅ |
| 8 | Threat Intel Feed + Device Manager + Voice Commands | ✅ |

---

## 📄 License
MIT — see [LICENSE](LICENSE)
