"""
Sentiment Classification Service
Wraps distilbert-base-uncased-finetuned-sst-2-english in a FastAPI service.
"""

import time
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.model import SentimentModel
from app.logger import get_structured_logger
from app.drift import DriftMonitor

# ── Structured logger ─────────────────────────────────────────────────────────
logger = get_structured_logger("sentiment_service")

# ── Globals ───────────────────────────────────────────────────────────────────
model: SentimentModel = None
drift_monitor: DriftMonitor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, drift_monitor
    logger.info("startup", message="Loading model...")
    model = SentimentModel()
    drift_monitor = DriftMonitor()
    logger.info("startup", message="Model ready.")
    yield
    logger.info("shutdown", message="Service shutting down.")


app = FastAPI(
    title="Sentiment Classification Service",
    description="Production sentiment API with logging, drift monitoring, and CI-gated retraining.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, description="Text to classify")
    request_id: str | None = Field(default=None, description="Optional caller-supplied idempotency key")

    @field_validator("text")
    @classmethod
    def strip_text(cls, v):
        return v.strip()


class PredictResponse(BaseModel):
    request_id: str
    label: str
    score: float
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    version: str


class DriftResponse(BaseModel):
    drift_detected: bool
    summary: dict


# ── Middleware: request/response audit log ────────────────────────────────────
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=elapsed_ms,
        client=request.client.host if request.client else "unknown",
    )
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness probe — used by Docker HEALTHCHECK and Render."""
    return HealthResponse(
        status="ok",
        model_name=model.model_name,
        version=app.version,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest):
    """Run sentiment classification on the provided text."""
    req_id = payload.request_id or str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        result = model.predict(payload.text)
    except Exception as exc:
        logger.error(
            "predict_error",
            request_id=req_id,
            text_length=len(payload.text),
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    timestamp = datetime.utcnow().isoformat() + "Z"

    # ── Structured prediction log (queryable for post-mortem) ────────────────
    logger.info(
        "prediction",
        request_id=req_id,
        text=payload.text,
        text_length=len(payload.text),
        label=result["label"],
        score=round(result["score"], 6),
        latency_ms=latency_ms,
        timestamp=timestamp,
        model_name=model.model_name,
    )

    # ── Feed drift monitor ────────────────────────────────────────────────────
    drift_monitor.record(payload.text)

    return PredictResponse(
        request_id=req_id,
        label=result["label"],
        score=round(result["score"], 4),
        latency_ms=latency_ms,
        timestamp=timestamp,
    )


@app.get("/drift", response_model=DriftResponse, tags=["ops"])
def drift_status():
    """Returns current drift statistics vs baseline."""
    report = drift_monitor.report()
    return DriftResponse(
        drift_detected=report["drift_detected"],
        summary=report,
    )


@app.post("/drift/reset", tags=["ops"])
def drift_reset():
    """Reset drift window — call after every model swap."""
    drift_monitor.reset()
    logger.info("drift_reset", message="Drift window reset by operator.")
    return {"status": "drift window reset"}