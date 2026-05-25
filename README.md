# Sentiment Classification Service
### Case 9 — Model Serving Lite

A production-ready sentiment analysis API wrapping DistilBERT, with structured logging, drift monitoring, and a CI-gated retraining pipeline.

---

## Live API

> **Base URL:** `https://your-service.onrender.com`
> *(Replace with your Render URL after deployment)*

---

## Quick Start (Local)

```bash
# 1. Create and activate virtual environment
python -m venv myenv
myenv\Scripts\activate        # Windows
source myenv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

### Health check
```bash
# curl (Linux/Mac)
curl http://localhost:8000/health

# PowerShell (Windows)
Invoke-WebRequest -Uri http://localhost:8000/health -Method GET |
  Select-Object -ExpandProperty Content
```

Response:
```json
{"status": "ok", "model_name": "distilbert-base-uncased-finetuned-sst-2-english", "version": "1.0.0"}
```

### Predict sentiment
```bash
# curl (Linux/Mac)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Absolutely love this product!"}'

# PowerShell (Windows)
Invoke-WebRequest -Uri http://localhost:8000/predict `
  -Method POST `
  -Headers @{"Content-Type" = "application/json"} `
  -Body '{"text": "Absolutely love this product!"}' |
  Select-Object -ExpandProperty Content
```

Response:
```json
{
  "request_id": "3f8a2b1c-...",
  "label": "POSITIVE",
  "score": 0.9998,
  "latency_ms": 42.3,
  "timestamp": "2024-06-01T14:22:05Z"
}
```

### Predict with custom request ID (for tracing)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible. Broke on day one.", "request_id": "order-789-complaint"}'
```

### Drift status
```bash
curl http://localhost:8000/drift
```

### Reset drift window (after model swap)
```bash
curl -X POST http://localhost:8000/drift/reset
```

---

## Docker

```bash
# Build
docker build -t sentiment-service:ci .

# Run
docker run -d --name svc -p 8000:8000 sentiment-service:ci

# Wait for model load (~60-90 seconds)
sleep 90
curl http://localhost:8000/health

# View logs
docker logs svc

# Stop
docker stop svc && docker rm svc
```

---

## Retrain Pipeline

The CI pipeline automatically retrains and gates the model when `data/train.csv` changes in a PR.

### Run locally

```bash
# Should PASS (low baseline)
python retrain/retrain.py \
  --train_file data/train.csv \
  --eval_file data/eval.csv \
  --output_dir models/candidate \
  --baseline_f1 0.80 \
  --min_f1_delta -0.01 \
  --epochs 1

# Should FAIL (impossibly high baseline)
python retrain/retrain.py \
  --train_file data/train.csv \
  --eval_file data/eval.csv \
  --output_dir models/candidate \
  --baseline_f1 0.999 \
  --min_f1_delta -0.01 \
  --epochs 1
```

### Trigger CI gate via PR

```bash
git checkout -b add-training-data
# Edit data/train.csv
git add data/train.csv
git commit -m "feat: add new training examples"
git push origin add-training-data
# Then open a Pull Request on GitHub
```

---

## Repository Structure

```
sentiment-service/
├── app/
│   ├── main.py        ← FastAPI routes, middleware
│   ├── model.py       ← HuggingFace pipeline wrapper
│   ├── logger.py      ← Structured JSON logger
│   └── drift.py       ← Drift monitor (length + vocab novelty)
├── retrain/
│   └── retrain.py     ← Fine-tune + F1 gate script
├── data/
│   ├── train.csv      ← Training data (PR here to trigger retrain)
│   └── eval.csv       ← Held-out evaluation set (never train on this)
├── .github/
│   └── workflows/
│       └── ci.yml     ← Three CI jobs: test, retrain-gate, docker-build
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── README.md
└── WRITEUP.md
```

---

## Deployment (Render.com)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service → Connect repo
3. Set Runtime to **Docker**
4. Add environment variable: `BASELINE_F1=0.92`
5. Deploy — your live URL will be `https://<your-service>.onrender.com`

> **Note:** Render free tier has a ~50 second cold start and spins down after 15 minutes of inactivity.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```