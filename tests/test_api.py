import requests

BASE = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200

def test_predict():
    payload = {"text": "This movie was amazing"}

    r = requests.post(
        f"{BASE}/predict",
        json=payload,
        timeout=5
    )

    assert r.status_code == 200
