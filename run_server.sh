#!/usr/bin/env bash
# Starts the API on port 8000. Loads variables from .env if it exists.
set -e
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${QA_API_TOKEN:?Set QA_API_TOKEN (in .env or in the terminal)}"
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
