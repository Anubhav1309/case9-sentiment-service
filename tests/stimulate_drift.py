import requests

BASE = "https://anubhav130-sentiment.hf.space"

# Phase 1: send 200 short normal requests to build baseline
print("Building baseline (200 requests)...")
for i in range(200):
    requests.post(f"{BASE}/predict", json={"text": "good product loved it"})

print("Baseline built. Checking drift status...")
print(requests.get(f"{BASE}/drift").json())

# Phase 2: send 50 very long weird-vocab requests to trigger drift
print("\nSending drifted requests (long medical text)...")
long_text = "The patient presented with acute exacerbation of chronic obstructive pulmonary disease requiring immediate bronchodilator therapy and corticosteroid administration." * 3
for i in range(50):
    requests.post(f"{BASE}/predict", json={"text": long_text})

print("\nDrift status after shift:")
print(requests.get(f"{BASE}/drift").json())

# Phase 3: reset and verify
requests.post(f"{BASE}/drift/reset")
print("\nAfter reset:")
print(requests.get(f"{BASE}/drift").json())
