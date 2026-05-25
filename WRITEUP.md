# How Would I Know This Model Is Failing Before Customers Do?

## The Four-Signal Early Warning System

### Signal 1 — Prediction Confidence Histogram (leading indicator)

The model returns a `score` field — a softmax probability. A healthy model on in-distribution text clusters near 0.99. When input distribution shifts, scores drift toward 0.5–0.7 (the model becomes "unsure") *before* labels visibly flip.

**How to catch it:** Alert on `p50(score) < 0.85` over a 15-minute window. This fires before a single customer complains.

```
Alert: avg(score) < 0.85 for 15 minutes → PagerDuty
```

This is a **leading indicator** — confidence drops before accuracy drops.

---

### Signal 2 — Drift Monitor (built-in, model-unaware)

`GET /drift` returns `drift_detected: true` when incoming text diverges from baseline on two signals:

| Signal | Method | Threshold |
|--------|--------|-----------|
| Text length shift | Z-score of mean length vs baseline | > 2.5σ |
| Vocabulary novelty | % of tokens absent from baseline vocab | > 40% |

This fires when the **input** changes, even before output degrades. It's model-unaware — it doesn't need to know the correct label.

**Incident pattern:** If drift fires AND confidence drops simultaneously → almost certainly a real distribution shift (new product category, API client bug, seasonal language change).

**What drift means in practice:**
- **Length drift** → app is now sending bulk/API text instead of short user reviews
- **Vocab novelty** → users switched domain or jargon (e.g. new product line launched)

---

### Signal 3 — Request Latency (infrastructure health)

The service logs `latency_ms` per request via HTTP middleware. A spike in P99 latency (e.g. > 2 seconds) indicates the container is memory-swapping or the model pipeline is degraded.

```
Alert: p99(latency_ms) > 2000 for 5 minutes → Slack #ml-ops
```

This catches infrastructure problems before they become user-facing failures.

---

### Signal 4 — Error Rate (5xx responses)

Every response status code is logged by the audit middleware. A burst of 500s means the model pipeline threw an exception — usually because input exceeded `max_length=512` tokens or because an upstream schema change broke the request format.

```
Alert: count(status_code=500) / count(all_requests) > 1% → immediate page
```

---

## The Post-Mortem Workflow (debugging a bad prediction a week later)

1. Get the `request_id` from the customer ticket or support tool
2. Search logs: `grep "request_id-value" logs/service.log`
3. The log line contains the full original text, label, score, and timestamp
4. Replay the exact text: `POST /predict` with the recovered text
5. If replay gives the correct label → transient issue (container was mid-restart)
6. If replay gives the wrong label → model needs retraining; add the example to `data/train.csv` with the correct label, open a PR, CI gate runs automatically

Example log line to search:
```json
{
  "event": "prediction",
  "request_id": "order-789-complaint",
  "text": "Terrible. Broke on day one.",
  "label": "NEGATIVE",
  "score": 0.9991,
  "latency_ms": 38.4,
  "timestamp": "2024-06-01T14:22:05Z"
}
```

---

## Honest Gaps (free-tier deploy)

| Gap | Production Fix |
|-----|----------------|
| Logs lost on container restart | Ship logs to Datadog / CloudWatch / Loki |
| Drift baseline resets on restart | Persist baseline to Redis or S3 |
| No shadow traffic / A-B testing | Route 5% of traffic to candidate before full promote |
| No output label monitoring | Log predicted vs user-corrected labels if app has a feedback loop |
| No GPU → ~200ms latency | Use `onnxruntime` quantised model for 4x CPU speedup |
| Render free tier cold starts (~50s) | Use paid tier or Fly.io for always-on |

---

## Summary

The four signals form a layered defence:

```
Signal 1: Confidence drops     ← model is getting uncertain (leading)
Signal 2: Drift detected       ← inputs changed (model-unaware)
Signal 3: Latency spikes       ← infrastructure degraded
Signal 4: 5xx errors           ← hard failures
```

Each layer catches a different failure mode. Together they give enough signal to detect and diagnose a production issue before customers notice it.