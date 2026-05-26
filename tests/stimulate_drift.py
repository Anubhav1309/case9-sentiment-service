import requests
import json

BASE = "https://Anubhav130-sentiment.hf.space"

print("Step 1: Checking drift status before simulation...")
r = requests.get(f"{BASE}/drift")
print(json.dumps(r.json(), indent=2))

print("\nStep 2: Sending 200 normal short requests to build baseline...")
for i in range(200):
    requests.post(f"{BASE}/predict", json={"text": "good product loved it"})
    if i % 50 == 0:
        print(f"  {i}/200 done...")

print("\nStep 3: Checking drift after baseline built...")
r = requests.get(f"{BASE}/drift")
print(json.dumps(r.json(), indent=2))

print("\nStep 4: Sending 50 long unusual requests to trigger drift...")
long_text = "The patient presented with acute exacerbation of chronic obstructive pulmonary disease requiring immediate bronchodilator therapy and corticosteroid administration contraindicated by renal dysfunction." * 3
for i in range(50):
    requests.post(f"{BASE}/predict", json={"text": long_text})

print("\nStep 5: Checking drift after shift — should show drift_detected: true")
r = requests.get(f"{BASE}/drift")
print(json.dumps(r.json(), indent=2))

