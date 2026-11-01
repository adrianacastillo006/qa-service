# Extractive QA as a Service (SQuAD 2.0)

Fine-tuned `roberta-base` for extractive question answering with unanswerable
questions, served with FastAPI behind a bearer token and exposed through a public tunnel.
We also trained `distilbert-base-uncased` with the same setup for comparison
(validation F1 68.3 vs 83.7 for RoBERTa); its logs are in `logs/distilbert-base-uncased/`.

| Doc | Content |
|---|---|
| `REPORT.md` | Project report (required) |
| `GUIA_IMPLEMENTACION.md` | Step-by-step guide (Spanish) |
| `COMO_FUNCIONA.md` | How the system works + Q&A practice (Spanish) |
| `docs/qa_api_contract.yaml` | API contract |
| `docs/dry_run/` | Dry-run evidence: curl screenshots of tests 1–5 and `check_service.py` output through the tunnel |
| `logs/<base-model>/` | Training log, loss curve, threshold sweep |

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                   # offline tests, no model needed

pip install -r requirements-train.txt       # on a GPU machine
python train.py --base_model roberta-base --epochs 2 --logs_dir logs/roberta-base
                                            # -> model/, logs/roberta-base/{loss.png,threshold.json}

cp .env.example .env                        # set QA_API_TOKEN and TEAM_NAME
./run_server.sh                             # Windows: .\run_server.ps1
python scripts/check_service.py --url http://127.0.0.1:8000
ngrok http 8000
```

On Windows use `127.0.0.1` rather than `localhost` for local checks: `localhost`
tries IPv6 first and adds ~2 s to every request.

## Model weights

Weights are not in git. Download `model.zip` from the GitHub Release
[`model-v1`](../../releases/tag/model-v1) and unzip it in the repository root so that
`model/config.json` and `model/qa_meta.json` exist.
