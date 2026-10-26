# Extractive QA as a Service (SQuAD 2.0)

Fine-tuned `distilbert-base-uncased` for extractive question answering with unanswerable
questions, served with FastAPI behind a bearer token and exposed through a public tunnel.

| Doc | Content |
|---|---|
| `REPORT.md` | Project report (required) |
| `GUIA_IMPLEMENTACION.md` | Step-by-step guide (Spanish) |
| `COMO_FUNCIONA.md` | How the system works + Q&A practice (Spanish) |
| `docs/qa_api_contract.yaml` | API contract |

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                   # offline tests, no model needed

pip install -r requirements-train.txt       # on a GPU machine
python train.py --epochs 2                  # -> model/, logs/loss.png, logs/threshold.json

cp .env.example .env                        # set QA_API_TOKEN and TEAM_NAME
./run_server.sh                             # Windows: .\run_server.ps1
python scripts/check_service.py --url http://localhost:8000
ngrok http 8000
```

Model weights are not in git: TODO link to the release / shared folder with `model.zip`.
