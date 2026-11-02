#!/usr/bin/env bash
# Opens the public ngrok tunnel to port 8000. Loads variables from .env if it exists.
# With NGROK_DOMAIN set (the free static dev domain of the ngrok account) the public URL never changes.
set -e
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -n "${NGROK_DOMAIN:-}" ]; then
  exec ngrok http 8000 --url="https://${NGROK_DOMAIN}"
else
  exec ngrok http 8000
fi
