# Project Report — Extractive QA as a Service

<!-- Replace every TODO with your real values before tagging `submission`.
     Do not leave any section empty: an empty section = report not delivered. -->

## 0. Team

| Field | Value |
|---|---|
| Team id | `TODO team-XX` |
| Members | TODO Name 1, Name 2, Name 3, Name 4, (Name 5) |
| Repository URL | TODO |
| `submission` commit hash | TODO (`git rev-parse --short submission`) |
| Public service URL used at the dry run | TODO `https://<name>.ngrok-free.app` |
| AI tools used (which, and for what) | Claude: initial scaffolding of `qa_engine.py`, `app.py`, `train.py` and offline tests; every module was then read, tested and adjusted by the team. TODO: add any other tool you used. |

## 1. Model choice and justification

- **Base checkpoint** (exact Hugging Face id): `distilbert-base-uncased` (66,362,880 parameters as reported by `/info`).
- **Alternatives considered:**
  - `roberta-base` (125 M): typically several F1 points better on SQuAD 2.0, but roughly 2× the CPU latency per window and 2× the size.
  - `microsoft/deberta-v3-base` (184 M): state of the art among base encoders, but the slowest on CPU and heavier to fine-tune in our 1.5-week window.
- **Trade-off:** the service runs CPU-only on a laptop behind a tunnel, with a 5 s per-request timeout and a 150 s cap for 50 items (3 s average budget including network). A 5000-character context becomes ~4–5 windows of 384 tokens; DistilBERT processes them in a fraction of a second, leaving most of the budget for tunnel latency. Since the grading reference is a DistilBERT model and points are capped at the reference, a larger model would add risk (timeouts) for no extra points.

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | TODO (default: full train, 130,319 questions → TODO features after windowing) |
| Validation data (what you validated on, how many examples) | Official SQuAD 2.0 dev set, shuffled: 2,000 questions for threshold tuning and EM/F1, the next 1,000 for validation loss |
| Epochs | TODO (default 2) |
| Learning rate and schedule (warmup?) | 3e-5, linear decay, 10 % warmup, weight decay 0.01 |
| Batch size | TODO (default 16) |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | TODO (e.g. NVIDIA T4, fp16, XX min — from `logs/threshold.json` → `training`) |

- **Loss curve:** [`logs/loss.png`](logs/loss.png) (training loss every 50 steps, validation loss every 500 steps; raw values in `logs/train_log.json`).
- **Validation results** (2,000 dev questions, tuned threshold; from `logs/threshold.json` → `best`):

| Subset | EM | F1 |
|---|---|---|
| All | TODO | TODO |
| Answerable | TODO | TODO |
| Unanswerable (accuracy of `null`) | TODO | — |

## 3. No-answer threshold

- **Decision rule** (`QAEngine.decide`): we compare the no-answer score `start_logit[CLS] + end_logit[CLS]` (minimum over windows) with the best valid span score `start_logit[i] + end_logit[j]` (best over all windows, span inside the context, `j ≥ i`, at most 30 tokens). We return `null` if `null_score − best_span_score > threshold`.
- **How the value was chosen:** `tune_threshold` in `train.py` runs the same `QAEngine.raw_scores` used by the API on 2,000 dev questions, stores `diff = null_score − best_span_score` per question, and sweeps every threshold between observed values, keeping the one that maximises overall F1. **Final value: TODO** (saved in `model/qa_meta.json`).
- **Too low / too high:** a lower threshold returns `null` more often — accuracy on unanswerable questions rises while answerable questions are lost; a higher threshold almost never returns `null`, which fails the ≈25 % unanswerable items of the hidden set. TODO: quote two points of the sweep in `logs/threshold.json` (e.g. threshold −4 and +4) to show this.

## 4. Serving architecture

- **Framework and server:** FastAPI + uvicorn, 1 worker (one copy of the model in RAM).
- **Model loading:** at startup a background thread (`load_engine`) loads tokenizer and weights from `MODEL_DIR` and runs one warm-up prediction. Until it finishes, `/health` returns `503 {"status":"loading"}`.
- **Request flow:** `/predict` → bearer-token check → JSON parsing and validation with a strict Pydantic model (types, lengths, no extra fields; errors return `400 bad_request`) → `QAEngine.predict` in a thread pool behind a lock → the question/context pair is split into 384-token windows with a 128-token stride, so long contexts are fully covered → all windows go through the model in one batch → best span vs. `[CLS]` score → character offsets via `offset_mapping` → the answer is sliced from the original context. A final check enforces `context[start:start+len(answer)] == answer` (otherwise `null`). Unexpected errors return `500 internal_error`.
- **Access control:** `is_authorized` in `app.py` reads `QA_API_TOKEN` from the environment and compares the whole `Authorization` header with `Bearer <token>` using `hmac.compare_digest` (exact, case-sensitive, constant time). If the variable is unset, every protected request is rejected. The token lives only in an untracked `.env` file.
- **Tunnel:** TODO ngrok (`ngrok http 8000`); backup: Cloudflare Quick Tunnel (`cloudflared tunnel --url http://localhost:8000`).
- **How to run** from a clean machine:

```bash
git clone <repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download model.zip from the release and unzip it so that ./model/config.json exists
export QA_API_TOKEN=...          # never committed
export TEAM_NAME=team-XX
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
python scripts/check_service.py --url http://localhost:8000
```

## 5. Results on the public sample set

Output of `python scripts/check_service.py --url <tunnel-url>` (TODO: replace with your run).

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` / `16 attention heads` (383) | TODO | TODO | TODO |
| sample-02 | `3.5 days` (437) | TODO | TODO | TODO |
| sample-03 | `null` | TODO | TODO | TODO |
| sample-04 | `null` | TODO | TODO | TODO |
| sample-05 | `50,257` / `50,257 tokens` (3157) | TODO | TODO | TODO |

TODO: comment on any wrong answer. Typical risks: sample-04 is adversarial (the context mentions "GPU memory requirement by 3 times", but for GPT-3, not BERT-base), so a model that matches keywords answers `3 times` instead of `null`; sample-05 only works if the answer in the last window is found, which our windowing handles.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 66,362,880 |
| Size on disk | TODO (≈255 MB for `model.safetensors`) |
| Mean latency per `/predict` request on your laptop (CPU) | TODO ms local / TODO ms through the tunnel |
| Laptop CPU and RAM | TODO |

## 7. Problems encountered and how you fixed them

TODO: write your own; these are examples of problems this code base addresses — keep only what really happened to you and cite your commits.

1. **Problem:** Requests without a token but with an invalid body returned 400 instead of 401. **Cause:** FastAPI validates the body before running header dependencies. **Fix:** `/predict` reads the raw request, checks the token first, then validates with Pydantic. **Commit:** `TODO`
2. **Problem:** Long contexts (sample-05, >512 tokens) were truncated and the answer was lost. **Cause:** default tokenization cuts at `max_seq_length`. **Fix:** windowing with `return_overflowing_tokens=True`, `stride=128`, and best-span search across windows. **Commit:** `TODO`
3. **Problem:** TODO (e.g. too many `null` answers with threshold 0). **Cause:** … **Fix:** threshold tuned on 2,000 dev questions. **Commit:** `TODO`

## 8. Decoder investigation

TODO: tie each point to a failure you actually observed.

- **Yes/no questions** (e.g. "Does BERT use next sentence prediction?"): the answer "yes" is not a span of the context, so our encoder can only return a related phrase or `null`. A decoder (e.g. a small instruction-tuned GPT-style model) or an encoder–decoder like T5 can generate "yes". Cost: autoregressive decoding needs one forward pass per output token, so latency on CPU grows from ~0.1 s to seconds for models of ≥1 B parameters, and free text breaks the `context[start:...] == answer` contract, so we would need to map the output back to the context or change the evaluation.
- **Non-contiguous answers** (e.g. "Which two objectives were used in pre-training?" → "masked language modeling and next sentence prediction" happens to be contiguous in sample-01, but lists split across sentences are not): a generator can join pieces; our model picks one span. Cost: same as above plus risk of hallucinated text that is not in the context.
- **Long contexts:** our windows see 384 tokens at a time and score each window independently, so evidence spread over two windows is never combined. Decoders with long context windows (8k+ tokens) read the whole 5,000-character context at once. Cost: attention is quadratic in length and the model is much larger; fine-tuning would need LoRA (Hu et al., 2021) to fit on our hardware.
- **Unanswerable questions:** a decoder can be prompted to say "not in the text", but calibrating that behaviour is harder than our single threshold on `[CLS]` vs. span scores.

## 9. Known limitations

- English only: other languages produce low-quality spans (the tokenizer was trained on English).
- One answer span per question; answers longer than 30 tokens are never returned.
- No yes/no, counting or arithmetic answers that are not literally in the text.
- Questions longer than 128 tokens are truncated before windowing.
- Requests are processed one at a time (lock); many concurrent clients would queue and could exceed the 5 s timeout.
- The service depends on the laptop and the tunnel staying up; if the tunnel URL changes it must be re-sent.
- The `score` is a sigmoid of the margin to the threshold, not a calibrated probability.
