"""
demo_web.py - Small web page to try the running service in a browser (for the presentation).

  python scripts/demo_web.py                              # http://127.0.0.1:8080, calls the public URL
  python scripts/demo_web.py --url http://127.0.0.1:8000  # calls the local service

Listens only on 127.0.0.1, so it is reachable from this laptop only. The page never sees the
token: this script reads QA_API_TOKEN from .env and adds it when it forwards /predict.

The "En vivo" panel lists every /predict that came in through the tunnel (the instructor's
harness included), read from ngrok's local inspector API. Only the JSON bodies are passed on,
never the request headers, so the Authorization token cannot end up on the projector.
"""
import argparse
import base64
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("demo_web.html")


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def body_json(raw_b64):
    """JSON body of a raw HTTP message from the ngrok inspector, or {} if there is none."""
    try:
        raw = base64.b64decode(raw_b64)
        return json.loads(raw.split(b"\r\n\r\n", 1)[1] or b"{}")
    except (ValueError, IndexError):
        return {}


def live_requests(inspect_url, limit=50):
    """The latest /predict calls seen by ngrok, without any header."""
    data = requests.get(f"{inspect_url}/api/requests/http", params={"limit": limit}, timeout=3).json()
    items = []
    for r in data.get("requests", []):
        if r["request"]["method"] != "POST" or not r["request"]["uri"].startswith("/predict"):
            continue
        req, resp = body_json(r["request"]["raw"]), body_json(r.get("response", {}).get("raw", ""))
        rid = str(req.get("id", ""))
        items.append({
            "time": r["start"], "ms": round(r["duration"] / 1e6),
            "status": r.get("response", {}).get("status_code"),
            "id": rid, "question": req.get("question"), "context": req.get("context"),
            "answer": resp.get("answer"), "start": resp.get("start"), "score": resp.get("score"),
            "error": resp.get("error"), "ours": rid.startswith(("demo-", "ask-")),
        })
    return items


def make_handler(base, token, inspect_url):
    auth = {"Authorization": f"Bearer {token}"}

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                data = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/info":
                try:
                    health = requests.get(f"{base}/health", timeout=5).json()
                    info = requests.get(f"{base}/info", headers=auth, timeout=5).json()
                    self.send_json(200, {"url": base, "health": health, "info": info})
                except requests.RequestException as e:
                    self.send_json(502, {"url": base, "error": str(e)})
            elif self.path == "/api/live":
                try:
                    self.send_json(200, {"items": live_requests(inspect_url)})
                except (requests.RequestException, ValueError, KeyError) as e:
                    self.send_json(502, {"error": f"ngrok inspector not reachable: {e}"})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/predict":
                return self.send_json(404, {"error": "not found"})
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            t0 = time.perf_counter()
            try:
                r = requests.post(f"{base}/predict", data=body, timeout=10,
                                  headers={**auth, "Content-Type": "application/json"})
            except requests.RequestException as e:
                return self.send_json(502, {"error": str(e)})
            ms = round((time.perf_counter() - t0) * 1000)
            self.send_json(200, {"status": r.status_code, "ms": ms, "response": r.json()})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    load_env()
    domain = os.environ.get("NGROK_DOMAIN")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=f"https://{domain}" if domain else "http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--inspect", default="http://127.0.0.1:4040", help="ngrok inspector (web_addr)")
    args = ap.parse_args()
    token = os.environ.get("QA_API_TOKEN")
    if not token:
        raise SystemExit("Set QA_API_TOKEN first (in .env).")
    handler = make_handler(args.url.rstrip("/"), token, args.inspect.rstrip("/"))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Demo en http://127.0.0.1:{args.port}  ->  servicio {args.url}  (Ctrl+C para salir)")
    server.serve_forever()


if __name__ == "__main__":
    main()
