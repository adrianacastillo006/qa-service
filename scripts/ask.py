"""
ask.py - Talk to the running service from the terminal: paste a paragraph (one line), then ask questions.

  python scripts/ask.py                                   # public URL from NGROK_DOMAIN in .env
  python scripts/ask.py --url http://127.0.0.1:8000       # local only

Reads QA_API_TOKEN (and NGROK_DOMAIN) from the environment or from .env.
Commands at the question prompt: 'new' = paste another paragraph, 'quit' = exit.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
GREEN, DIM, BOLD, RESET = "\033[92m", "\033[2m", "\033[1m", "\033[0m"


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def read_context():
    print(f"\n{BOLD}PASO 1:{RESET} pega el parrafo en ingles (todo en una linea) y presiona Enter.")
    while True:
        context = input(f"{BOLD}Parrafo>{RESET} ").strip()
        if context:
            print(f"{GREEN}Parrafo guardado ({len(context)} caracteres).{RESET}")
            print(f"\n{BOLD}PASO 2:{RESET} escribe preguntas sobre ese parrafo, una por linea.")
            print(f"{DIM}   'new' = cambiar de parrafo   'quit' = salir{RESET}")
            return context


def main():
    load_env()
    domain = os.environ.get("NGROK_DOMAIN")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=f"https://{domain}" if domain else "http://127.0.0.1:8000")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    token = os.environ.get("QA_API_TOKEN")
    if not token:
        sys.exit("Set QA_API_TOKEN first (in .env).")
    if os.name == "nt":
        os.system("")  # enables ANSI colors in the Windows console

    info = requests.get(f"{base}/info", headers={"Authorization": f"Bearer {token}"}, timeout=10).json()
    print(f"Conectado a {base}  ->  {info.get('model')} ({info.get('params'):,} params)")

    context = read_context()
    n = 0
    while True:
        question = input(f"\n{BOLD}Pregunta>{RESET} ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "new":
            context = read_context()
            continue
        n += 1
        t0 = time.perf_counter()
        r = requests.post(f"{base}/predict", timeout=10,
                          headers={"Authorization": f"Bearer {token}"},
                          json={"id": f"ask-{n}", "context": context, "question": question})
        ms = (time.perf_counter() - t0) * 1000
        body = r.json()
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {body}")
            continue
        answer, start = body["answer"], body["start"]
        if answer is None:
            print(f"-> null (no esta en el parrafo)   score={body['score']:.2f}   {ms:.0f} ms")
        else:
            print(f"-> {GREEN}{answer}{RESET}   start={start}   score={body['score']:.2f}   {ms:.0f} ms")
            a, b = max(0, start - 60), min(len(context), start + len(answer) + 60)
            print(f"   {DIM}...{context[a:start]}{RESET}{GREEN}{answer}{RESET}{DIM}{context[start + len(answer):b]}...{RESET}")


if __name__ == "__main__":
    main()
