import requests
import json
BASE = "http://localhost:8000"

# Health
r = requests.get(f"{BASE}/health")
print("HEALTH:", json.dumps(r.json(), indent=2))

# Positive
r = requests.post(f"{BASE}/predict", json={"text": "Absolutely love this product!"})
print("POSITIVE:", json.dumps(r.json(), indent=2))

# Negative
r = requests.post(f"{BASE}/predict", json={
    "text": "Terrible quality, broke on day one.",
    "request_id": "test-001"
})
print("NEGATIVE:", json.dumps(r.json(), indent=2))

# Drift
r = requests.get(f"{BASE}/drift")
print("DRIFT:", json.dumps(r.json(), indent=2))

