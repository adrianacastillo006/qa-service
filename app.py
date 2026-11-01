"""
app.py - Extractive QA web service (implements docs/qa_api_contract.yaml).

Endpoints
  GET  /health   public     200 {"status":"ok"} | 503 {"status":"loading"}
  GET  /info     protected  team, model, params, max_seq_length, doc_stride
  POST /predict  protected  {"id","answer","start","score"}

Run:
  export QA_API_TOKEN=<token>        # never hard-code it
  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("qa-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
TEAM_NAME = os.environ.get("TEAM_NAME", "team-XX")
QA_DEVICE = os.environ.get("QA_DEVICE") or None   # "cpu" / "cuda"; default: cuda if available

# --------------------------------------------------------------------- state
state = {"engine": None, "load_error": None}
predict_lock = threading.Lock()   # one inference at a time -> predictable CPU latency


def load_engine() -> None:
    """Loads the model in a background thread so /health can answer 503 meanwhile."""
    try:
        from qa_engine import QAEngine
        t0 = time.time()
        engine = QAEngine(MODEL_DIR, device=QA_DEVICE)
        # first calls are slower; warm up with a short and a ~5000-char (multi-window) input
        engine.predict("Warm-up?", "Warm-up context.")
        engine.predict("Warm-up?", "Warm-up context. " * 300)
        state["engine"] = engine
        log.info("Model loaded from %s on %s in %.1fs (%d params, threshold=%.3f)",
                 MODEL_DIR, engine.device, time.time() - t0, engine.n_params, engine.threshold)
    except Exception as exc:  # keep the server alive; /health stays 503
        state["load_error"] = repr(exc)
        log.exception("Model failed to load")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not os.environ.get("QA_API_TOKEN"):
        log.warning("QA_API_TOKEN is not set: every protected request will get 401.")
    threading.Thread(target=load_engine, daemon=True).start()
    yield


app = FastAPI(title="Extractive QA Service", version="1.0", lifespan=lifespan)


# -------------------------------------------------------------------- errors
def error(status: int, code: str, detail: str | None = None, headers=None) -> JSONResponse:
    body = {"error": code}
    if detail:
        body["detail"] = detail
    return JSONResponse(status_code=status, content=body, headers=headers)


@app.exception_handler(RequestValidationError)
async def _validation_handler(_req, exc):
    return error(400, "bad_request", str(exc.errors()[0].get("msg", "invalid body")))


@app.exception_handler(StarletteHTTPException)
async def _http_handler(_req, exc):
    code = {401: "unauthorized", 500: "internal_error"}.get(exc.status_code, "bad_request")
    return error(exc.status_code, code, None if exc.status_code == 401 else str(exc.detail))


@app.exception_handler(Exception)
async def _unhandled_handler(_req, exc):
    log.exception("Unhandled error")
    return error(500, "internal_error")


# ------------------------------------------------------------ access control
def is_authorized(request: Request) -> bool:
    """Exact, case-sensitive check of 'Authorization: Bearer <QA_API_TOKEN>'."""
    expected = os.environ.get("QA_API_TOKEN", "")
    if not expected:                       # fail closed if the token is not configured
        return False
    header = request.headers.get("authorization", "")
    # compare_digest avoids leaking the token through response timing
    return hmac.compare_digest(header.encode(), f"Bearer {expected}".encode())


def unauthorized() -> JSONResponse:
    return error(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})


# ------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)   # contract: additionalProperties false
    id: str = Field(min_length=1, max_length=64)
    context: str = Field(min_length=1, max_length=5000)
    question: str = Field(min_length=1, max_length=500)


# ----------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    if state["engine"] is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}


@app.get("/info")
def info(request: Request):
    if not is_authorized(request):
        return unauthorized()
    eng = state["engine"]
    if eng is None:
        return error(503, "internal_error", "model is still loading")
    return {
        "team": TEAM_NAME,
        "model": eng.base_model,
        "params": eng.n_params,
        "max_seq_length": eng.max_seq_length,
        "doc_stride": eng.doc_stride,
    }


@app.post("/predict")
async def predict(request: Request):
    # 1) access control first: no token -> 401, whatever the body is
    if not is_authorized(request):
        return unauthorized()

    # 2) parse and validate the body ourselves -> clean 400 messages
    try:
        payload = await request.json()
    except Exception:
        return error(400, "bad_request", "body is not valid JSON")
    if not isinstance(payload, dict):
        return error(400, "bad_request", "body must be a JSON object")
    for field in ("id", "context", "question"):
        if field not in payload:
            return error(400, "bad_request", f"field '{field}' is required")
    try:
        req = PredictRequest(**payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first.get("loc", []))
        return error(400, "bad_request", f"{loc}: {first.get('msg')}")

    eng = state["engine"]
    if eng is None:
        return error(503, "internal_error", "model is still loading")

    # 3) inference (CPU-bound -> run in a worker thread, not in the event loop)
    t0 = time.perf_counter()
    try:
        from starlette.concurrency import run_in_threadpool

        def _run():
            with predict_lock:
                return eng.predict(req.question, req.context)

        result = await run_in_threadpool(_run)
    except Exception:
        log.exception("Inference failed for id=%s", req.id)
        return error(500, "internal_error")

    # 4) safety net: the contract invariant must hold, otherwise return null
    ans, start = result["answer"], result["start"]
    if ans is not None and req.context[start:start + len(ans)] != ans:
        log.error("Span/offset mismatch for id=%s; returning null", req.id)
        result = {"answer": None, "start": None, "score": 0.0}

    log.info("id=%s answer=%r %.0f ms", req.id, result["answer"], (time.perf_counter() - t0) * 1000)
    return {"id": req.id, **result}
