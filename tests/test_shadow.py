import requests
import random
import time

BASE = "https://anubhav130-case9-sentiment-mlops-service.hf.space"

texts = [
    "Amazing product loved it",
    "Worst experience ever",
    "Delivery was slow but food tasted good",
    "Terrible customer support",
    "Absolutely fantastic quality",
    "The packaging was damaged and disappointing",
    "Fast delivery and excellent support",
]

print("Starting shadow deployment test...\n")

for i in range(30):

    text = random.choice(texts)

    try:
        response = requests.post(
            f"{BASE}/predict",
            json={"text": text},
            timeout=60
        )

        print(f"\n========== Request {i+1} ==========")
        print("TEXT:", text)
        print("STATUS:", response.status_code)

        # safely parse JSON
        try:
            data = response.json()
            print("RESPONSE:", data)
        except Exception:
            print("Non-JSON response:")
            print(response.text)

    except requests.exceptions.Timeout:
        print(f"\nRequest {i+1} timed out")

    except requests.exceptions.ConnectionError:
        print(f"\nConnection failed on request {i+1}")

    except Exception as e:
        print(f"\nUnexpected error on request {i+1}: {e}")

    # small delay so HF Space doesn't overload
    time.sleep(1)

print("\nFinished sending requests.")

# ── Check shadow deployment results ─────────────────────
print("\nFetching shadow deployment report...\n")

try:
    shadow_response = requests.get(
        f"{BASE}/shadow",
        timeout=30
    )

    print("SHADOW STATUS:", shadow_response.status_code)
    print(shadow_response.json())

except Exception as e:
    print("Failed to fetch shadow report:", e)