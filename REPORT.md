# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `TODO team-XX` |
| Members | Miguel Ledezma, Marco Rosalez, Adriana Castillo, Juan Ramirez |
| Repository URL | https://github.com/adrianacastillo006/qa-service |
| `submission` commit hash | The commit tagged `submission` (a commit cannot contain its own hash: `git rev-parse --short submission`) |
| Public service URL used at the dry run | `https://dc6f-190-62-86-130.ngrok-free.app` (ngrok, random URL); evidence in [`docs/dry_run/`](docs/dry_run/). From now on the service uses the account's static ngrok domain `https://hippo-neat-yearly.ngrok-free.app`, which does not change between runs. |
| AI tools used (which, and for what) | Claude (Claude Code): scaffolding of `qa_engine.py`, `app.py`, `train.py` and the offline tests; running the two fine-tuning runs; prior-weighted threshold tuning; automating the dry-run screenshots; drafting this report. The team read, tested and adjusted every module. |

## 1. Model choice and justification

- **Base checkpoint** (exact Hugging Face id): `roberta-base` (124,056,578 parameters with the QA head, as reported by `/info`).
- **Alternatives considered:**
  - `distilbert-base-uncased` (66 M), trained with the identical script and hyperparameters: validation F1 **68.3** vs **83.7** for RoBERTa on the same 2,000 dev questions. It is ~2× faster, but the grading reference is a DistilBERT (0.70), so it leaves no margin on a hidden set written on non-SQuAD texts. Logs: [`logs/distilbert-base-uncased/`](logs/distilbert-base-uncased/).
  - `microsoft/deberta-v3-base` (184 M): stronger, but ~1.5–2× slower than RoBERTa on CPU and slower to fine-tune; RoBERTa already clears the reference.
- **Trade-off:** the 5 s timeout includes ~0.75 s of tunnel round trip. The worst input the contract allows (5,000-character context + 500-character question = 6 windows) takes ~2 s on the laptop CPU and ~0.2 s on its GPU. RoBERTa fits even CPU-only and gains ~15 F1 points.

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | Full SQuAD 2.0 train: 130,319 questions → 131,821 features after windowing |
| Validation data (what you validated on, how many examples) | SQuAD 2.0 dev, shuffled (seed 42): 2,000 questions (979 answerable, 1,021 unanswerable) for threshold tuning and EM/F1; the next 1,000 for validation loss |
| Epochs | 2 (16,478 steps) |
| Learning rate and schedule (warmup?) | 3e-5, linear decay, 10 % warmup, weight decay 0.01, fp16 |
| Batch size | 16 |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | NVIDIA RTX 4080 Laptop GPU (12 GB): 62 min (RoBERTa), 35 min (DistilBERT) |

- **Loss curve:** [`logs/roberta-base/loss.png`](logs/roberta-base/loss.png) (raw values in `train_log.json`, console in `train_stdout.txt`). Validation loss falls from 1.52 to a minimum of 0.83 and stays ~0.90 in epoch 2 while training loss reaches 0.61: mild overfitting, so more epochs would not help. DistilBERT: [`logs/distilbert-base-uncased/loss.png`](logs/distilbert-base-uncased/loss.png) (min 1.21).
- **Validation results** (2,000 dev questions; [`logs/roberta-base/threshold.json`](logs/roberta-base/threshold.json)):

| Model / threshold | EM all | F1 all | EM answerable | F1 answerable | Unanswerable accuracy |
|---|---|---|---|---|---|
| RoBERTa, −2.09 (tuned on raw dev, 51 % unanswerable) | 80.8 | 83.7 | 75.5 | 81.4 | 85.9 |
| **RoBERTa, +1.33 (tuned for 25 % unanswerable, served)** | 79.4\* | 84.3\* | 79.9 | 86.4 | 78.0 |
| DistilBERT, −1.25 (raw dev) | 66.1 | 68.3 | 62.4 | 66.9 | 69.5 |
| DistilBERT, +2.46 (25 % unanswerable) | 67.0\* | 71.7\* | 73.0 | 79.4 | 48.8 |

\* weighted so unanswerable questions count as 25 % of the set, like the hidden test set.

## 3. No-answer threshold

- **Decision rule** (`QAEngine.decide`): `null_score = start_logit[CLS] + end_logit[CLS]` (minimum over windows; `<s>` in RoBERTa). The span score is the best valid `start_logit[i] + end_logit[j]` over all windows (inside the context, `j ≥ i`, ≤ 30 tokens). We return `null` if `null_score − best_span_score > threshold`.
- **How we chose it:** `tune_threshold` (`train.py`) runs the API's own `QAEngine.raw_scores` on the 2,000 dev questions and sweeps every threshold between the observed `diff` values for the best F1. The dev set is 51 % unanswerable but the hidden set ≈25 %, so we weight questions to a 25 % prior (`--target_unanswerable 0.25`). The optimum moves from −2.09 to **+1.33** (stored in `model/qa_meta.json`); `--tune_only` re-tunes without retraining.
- **Too low / too high** (weighted sweep): at −4 the service over-abstains (unanswerable accuracy 88.6 %, answerable F1 76.1, overall F1 79.2). At +4 it under-abstains (answerable F1 88.1, unanswerable accuracy 70.5 %, overall 83.7). +1.33 gives 84.3.

## 4. Serving architecture

- **Framework and server:** FastAPI + uvicorn, 1 worker.
- **Model loading:** at startup a background thread (`load_engine`) loads tokenizer and weights from `MODEL_DIR` on the GPU if available, else the CPU (`QA_DEVICE=cpu` forces it). It then runs a short and a ~5,000-character warm-up prediction. Until then `/health` returns `503 {"status":"loading"}`.
- **Request flow:** token check → JSON parsing and strict Pydantic validation (types, lengths, no extra fields; else `400 bad_request`) → `QAEngine.predict` in a thread pool behind a lock. The question (≤ 128 tokens) and context are split into 384-token windows with stride 128, all run in one batch. Best span vs `[CLS]` → character offsets via `offset_mapping` → the answer is sliced from the original context. A final check enforces `context[start:start+len(answer)] == answer` (else `null`); unexpected errors return `500 internal_error`.
- **Access control:** `is_authorized` (`app.py`) compares the whole `Authorization` header with `Bearer <QA_API_TOKEN>` using `hmac.compare_digest` (exact, case-sensitive, constant time), on `/info` and `/predict`. If the variable is unset, every request is rejected. The token lives only in the git-ignored `.env`; for the dry run we generated a random development token, as the instructor indicated.
- **Tunnel:** ngrok on the account's static dev domain (`run_tunnel.ps1` → `ngrok http 8000 --url=https://hippo-neat-yearly.ngrok-free.app`, domain read from `NGROK_DOMAIN` in `.env`); backup: `cloudflared tunnel --url http://localhost:8000`.
- **How to run** from a clean machine:

```bash
git clone https://github.com/adrianacastillo006/qa-service && cd qa-service
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# download model.zip from the GitHub Release model-v1 and unzip it so that ./model/config.json exists
cp .env.example .env      # set QA_API_TOKEN=<token> and TEAM_NAME=<team id>; never committed
./run_server.sh           # Windows: .\run_server.ps1  (uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1)
python scripts/check_service.py --url http://127.0.0.1:8000
./run_tunnel.sh           # Windows: .\run_tunnel.ps1  (ngrok http 8000 --url=https://$NGROK_DOMAIN)
```

## 5. Results on the public sample set

`python scripts/check_service.py --url https://dc6f-190-62-86-130.ngrok-free.app`: all contract checks passed through the tunnel ([output](docs/dry_run/check_service_tunnel.txt); screenshots of tests 1–5 in [`docs/dry_run/`](docs/dry_run/)).

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `16` / `16 attention heads` (383) | `16` | 383 | Yes |
| sample-02 | `3.5 days` (437) | `3.5 days` | 437 | Yes |
| sample-03 | `null` | `null` | `null` | Yes |
| sample-04 | `null` | `3 times` | 310 | No |
| sample-05 | `50,257` / `50,257 tokens` (3157) | `50,257` | 3157 | Yes |

**sample-04** is adversarial. The text says LoRA cuts "the GPU memory requirement by 3 times" relative to *GPT-3 175B*; the question asks about *BERT-base*. The model matches the wording and ignores the entity mismatch. Its margin is −4.87: only a threshold below that would yield `null`, costing ~10 answerable F1 points on validation. So it is a comprehension error, not a threshold problem, and we did not tune on it. **sample-05** (3 windows) is answered from the last window, confirming the windowing. DistilBERT also scored 4/5, but got sample-03 only by a hair (margin 2.60 vs threshold 2.46).

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 124,056,578 |
| Size on disk | 496 MB (`model.safetensors`, fp32) |
| Mean latency per `/predict` request on your laptop (CPU) | 372 ms on the sample set (130–160 ms per window-1 item, ~1.4 s for sample-05); ~2 s for the 6-window worst case. Served on the GPU: 50–260 ms server-side; ~0.8–1.1 s mean through ngrok (50 items ≈ 40–57 s, cap 150 s) |
| Laptop CPU and RAM | Alienware x16 R1: Intel Core i9-13900HK (14 cores / 20 threads), 32 GB RAM, RTX 4080 Laptop GPU 12 GB |

## 7. Problems encountered and how you fixed them

1. **Problem:** `TrainingArguments(warmup_ratio=...)` failed. **Cause:** transformers 5 removed `warmup_ratio`; `warmup_steps` takes a float < 1 as a ratio. **Fix:** `warmup_steps=0.1`, verified in `get_warmup_steps`. **Commit:** `d045b80`
2. **Problem:** `import datasets` failed with `DLL load failed ... An Application Control policy has blocked this file`. **Cause:** Windows Smart App Control blocks pandas 3.1's native DLLs. **Fix:** pin `pandas<3`. **Commit:** `41b16f0`
3. **Problem:** the threshold tuned on raw dev F1 over-abstained: answerable F1 only 66.9 (DistilBERT) / 81.4 (RoBERTa). **Cause:** dev is 51 % unanswerable, the hidden set ≈25 %. **Fix:** prior-weighted tuning plus `--tune_only`. Answerable F1 rose to 79.4 / 86.4, and 25 %-weighted F1 went 67.6 → 71.7 / 82.5 → 84.3. **Commit:** `352d89c`
4. **Problem:** the first 5,000-character request on CPU took 4.2 s (later ones ~1.9 s), at the 5 s timeout once tunnel latency is added; local measurements were also ~2 s too high. **Cause:** first-call initialisation for large shapes (the warm-up input was tiny), and on Windows `localhost` tries IPv6 first. **Fix:** multi-window warm-up, GPU serving when present (`QA_DEVICE` forces CPU), and `127.0.0.1` for local checks. **Commits:** `8c94bf6`, `df81c24`

## 8. Decoder investigation

- **Adversarial unanswerables (sample-04).** Our encoder scores spans by lexical/positional fit, so "3 times" beats `[CLS]` by 4.9 logits despite the GPT-3 vs BERT-base mismatch. An instruction-tuned decoder can be prompted to check that the entity matches before answering, or to say "not stated". *Cost:* a 1–3 B decoder is 10–25× larger and decodes token by token. A 5,000-character prompt would take seconds on our CPU and break the 5 s budget without the GPU, and it needs its own abstention calibration.
- **Prior shift (Problem 3).** Our abstention is one number we re-tune in seconds. A decoder that writes "no answer" bakes the prior into its fine-tuning data or prompt, and fixing it means re-fine-tuning plus re-evaluation. Here the encoder is cheaper.
- **Long contexts (sample-05: 3 windows; worst case 6).** Windows are scored independently, so evidence split across windows is never combined. A long-context decoder (8 k+ tokens) or LongT5 reads all 5,000 characters at once. *Cost:* quadratic attention and a much larger model; fine-tuning would need LoRA (Hu et al., 2021) on our 12 GB GPU.
- **Non-span answers.** Yes/no questions or lists spread over sentences cannot be one contiguous span; T5/GPT-style generators can write them. *Cost:* free text breaks `context[start:start+len(answer)] == answer`, so it must be mapped back to the context or the evaluation changed (exact match becomes unreliable).

## 9. Known limitations

- English only; one contiguous span per question, at most 30 tokens; no yes/no or computed answers.
- Questions over 128 tokens are truncated before windowing.
- The threshold assumes ≈25 % unanswerable; a very different share makes the service over- or under-abstain (Section 3).
- One request at a time (lock): concurrent clients queue and, on CPU, could exceed 5 s.
- CPU-only worst case is ~2 s; with a busy CPU and a slow tunnel, a 5,000-character request could approach the timeout.
- The laptop and tunnel must stay up; a new tunnel URL must be re-sent.
- `score` is a sigmoid of the margin to the threshold, not a calibrated probability.
