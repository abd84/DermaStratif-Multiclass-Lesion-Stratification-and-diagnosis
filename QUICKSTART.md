# DermaStratif — Quick Start

## Local

```bash
git clone https://github.com/abd84/Derma-Stratif-Inference.git
cd Derma-Stratif-Inference

make setup
make dev
```

Visit: **http://127.0.0.1:5001**

### Manual setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: set VISION_API_KEY
cd Application && FLASK_PORT=5001 FLASK_ENV=development python app.py
```

---

## Docker

```bash
docker-compose up --build
```

Visit: **http://localhost:5000**

---

## Requirements

- Python 3.9+
- ~2GB disk (models + deps)
- Optional: GPU (CUDA/MPS)

---

## Configuration (`.env`)

| Variable | Description |
|----------|-------------|
| `VISION_API_KEY` | Optional cloud vision API key |
| `FLASK_ENV` | `development` or `production` |
| `DEVICE` | `auto`, `cpu`, `cuda`, or `mps` |
| `MODEL_PATH` | Path to `best_model1_lora.pth` |

Without `VISION_API_KEY`, the app uses the local EfficientNet-LoRA model only.

---

## Models

| File | Role |
|------|------|
| `Saved Models/best_model1_lora.pth` | **Required** — deployed CNN fallback |
| `Saved Models/best_model_base.pth` | Optional base checkpoint |

---

## Commands

| Command | Purpose |
|---------|---------|
| `make setup` | Install deps, create `.env` |
| `make dev` | Dev server on port 5001 |
| `make prod` | Gunicorn on port 5000 |
| `make docker-up` | Start Docker stack |

---

## Docs

- [README.md](README.md) — overview
- [DEPLOYMENT.md](DEPLOYMENT.md) — production & troubleshooting
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributions

---

**Disclaimer:** Educational use only — not medical advice.
