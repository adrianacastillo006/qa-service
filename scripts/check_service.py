"""
check_service.py - Reproduces the instructor's checks against your service.

  python scripts/check_service.py --url http://localhost:8000
  python scripts/check_service.py --url https://<your-tunnel-url>

Reads the token from QA_API_TOKEN. Runs the 5 tests of the brief, the
public sample set (with the contract invariant check) and a latency summary.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import item_scores  # noqa: E402

TIMEOUT = 5.0


def show(ok, name, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--samples", default="data/public_sample.json")
    ap.add_argument("--repeat", type=int, default=3, help="repetitions for latency")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    token = os.environ.get("QA_API_TOKEN")
    if not token:
        sys.exit("Set QA_API_TOKEN first.")
    auth = {"Authorization": f"Bearer {token}"}
    all_ok = True

    print("== Contract tests ==")
    r = requests.get(f"{base}/health", timeout=TIMEOUT)
    all_ok &= show(r.status_code == 200 and r.json().get("status") == "ok", "1 /health no token", r.text)
    r = requests.get(f"{base}/info", timeout=TIMEOUT)
    all_ok &= show(r.status_code == 401 and r.json().get("error") == "unauthorized", "2 /info no token", str(r.status_code))
    r = requests.get(f"{base}/info", headers={"Authorization": "Bearer wrong"}, timeout=TIMEOUT)
    all_ok &= show(r.status_code == 401, "3 /info wrong token", str(r.status_code))
    r = requests.get(f"{base}/info", headers=auth, timeout=TIMEOUT)
    info = r.json() if r.ok else {}
    all_ok &= show(r.status_code == 200 and {"team", "model", "params"} <= info.keys(), "4 /info valid token", json.dumps(info))
    r = requests.post(f"{base}/predict", headers=auth, json={"id": "x", "context": "abc"}, timeout=TIMEOUT)
    all_ok &= show(r.status_code in (400, 422) and r.json().get("error") == "bad_request", "extra: missing field -> 400", r.text)
    r = requests.post(f"{base}/predict", json={"id": "x", "context": "a", "question": "b"}, timeout=TIMEOUT)
    all_ok &= show(r.status_code == 401, "extra: /predict no token -> 401", str(r.status_code))

    print("\n== Public sample set ==")
    items = json.loads(Path(args.samples).read_text())["items"]
    latencies, ems, f1s = [], [], []
    print(f"{'id':<10} {'expected':<22} {'answer':<22} {'start':>5} {'ms':>6}  ok")
    for it in items:
        body = {k: it[k] for k in ("id", "context", "question")}
        for rep in range(args.repeat):
            t0 = time.perf_counter()
            r = requests.post(f"{base}/predict", headers=auth, json=body, timeout=TIMEOUT)
            latencies.append((time.perf_counter() - t0) * 1000)
        res = r.json()
        valid = (r.status_code == 200 and res.get("id") == it["id"]
                 and {"id", "answer", "start", "score"} <= res.keys()
                 and ((res["answer"] is None) == (res["start"] is None))
                 and (res["answer"] is None
                      or it["context"][res["start"]:res["start"] + len(res["answer"])] == res["answer"]))
        golds = [a["text"] for a in it["answers"]]
        em, f1 = item_scores(res.get("answer"), golds) if valid else (0.0, 0.0)
        ems.append(em); f1s.append(f1)
        expected = golds[0] if golds else "null"
        print(f"{it['id']:<10} {expected:<22.22} {str(res.get('answer')):<22.22} "
              f"{str(res.get('start')):>5} {latencies[-1]:6.0f}  {'OK' if em == 1 else ('~' if f1 > 0 else 'X')}"
              f"{'' if valid else '  INVALID RESPONSE'}")
        all_ok &= valid

    print(f"\nEM={sum(ems)/len(ems):.2f}  F1={sum(f1s)/len(f1s):.2f}")
    print(f"Latency ms: mean={statistics.mean(latencies):.0f} "
          f"p95={sorted(latencies)[int(.95 * (len(latencies) - 1))]:.0f} max={max(latencies):.0f}")
    print(f"Projected time for 50 items: {statistics.mean(latencies) * 50 / 1000:.1f}s (cap 150s)")
    print("\nALL CONTRACT CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
