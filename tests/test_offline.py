"""
Offline tests: run without GPU, without torch and without the trained model.

  pytest -q

They check (1) the span/offset post-processing on the public sample set,
including the long context that needs several windows, and (2) the API
contract: auth, errors, response fields and the context[start:...] invariant.
A tiny tokenizer is built from the sample texts and the model logits are faked.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLES = json.loads((ROOT / "data/public_sample.json").read_text())["items"]

from qa_engine import QAEngine  # noqa: E402


# ------------------------------------------------------------------ helpers
def tiny_tokenizer(tmp_path):
    from transformers import BertTokenizerFast
    words = set()
    for it in SAMPLES:
        words |= set(re.findall(r"\w+|[^\w\s]", (it["context"] + " " + it["question"]).lower()))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(words)
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    return BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"), do_lower_case=True)


class FakeEngine(QAEngine):
    """QAEngine without a neural network: logits point at a chosen character span."""
    def __init__(self, tokenizer, max_seq_length=384, doc_stride=128, threshold=0.0):
        self.tokenizer = tokenizer
        self.max_seq_length, self.doc_stride, self.threshold = max_seq_length, doc_stride, threshold
        self.base_model, self.n_params = "fake-model", 123
        self.target = None  # (start_char, end_char) or None for "no answer"

    def _logits(self, enc):
        n, L = len(enc["input_ids"]), len(enc["input_ids"][0])
        s, e = np.zeros((n, L)), np.zeros((n, L))
        for w in range(n):
            s[w, 0] = e[w, 0] = 3.0                      # [CLS] = no-answer score 6
            if self.target is None:
                continue
            for t, (a, b) in enumerate(enc["offset_mapping"][w]):
                if enc.sequence_ids(w)[t] != 1:
                    continue
                if a == self.target[0]:
                    s[w, t] = 8.0
                if b == self.target[1]:
                    e[w, t] = 8.0
        return s, e


@pytest.fixture
def engine(tmp_path):
    return FakeEngine(tiny_tokenizer(tmp_path))


# ------------------------------------------------------- engine tests
@pytest.mark.parametrize("item", SAMPLES, ids=[i["id"] for i in SAMPLES])
def test_engine_on_public_samples(engine, item):
    if item["is_impossible"]:
        engine.target = None
    else:
        a = item["answers"][0]
        engine.target = (a["start"], a["start"] + len(a["text"]))
    out = engine.predict(item["question"], item["context"])
    if item["is_impossible"]:
        assert out["answer"] is None and out["start"] is None
    else:
        assert out["answer"] == item["answers"][0]["text"]
        assert out["start"] == item["answers"][0]["start"]
        assert item["context"][out["start"]:out["start"] + len(out["answer"])] == out["answer"]
    assert 0.0 <= out["score"] <= 1.0


def test_long_context_uses_several_windows(engine):
    long_item = next(i for i in SAMPLES if i["id"] == "sample-05")
    enc = engine._encode(long_item["question"], long_item["context"])
    assert len(enc["input_ids"]) > 1


def test_threshold_controls_null(engine):
    it = SAMPLES[0]
    a = it["answers"][0]
    engine.target = (a["start"], a["start"] + len(a["text"]))
    rs = engine.raw_scores(it["question"], it["context"])
    assert engine.decide(rs, threshold=0.0)["answer"] == a["text"]      # span 16 vs null 6
    assert engine.decide(rs, threshold=-20.0)["answer"] is None          # very strict -> null


# ---------------------------------------------------------- API tests
@pytest.fixture
def client(engine, monkeypatch):
    monkeypatch.setenv("QA_API_TOKEN", "dev-token-123")
    import app as app_module
    monkeypatch.setattr(app_module, "load_engine", lambda: None)
    from fastapi.testclient import TestClient
    with TestClient(app_module.app) as c:
        app_module.state["engine"] = engine
        yield c
    app_module.state["engine"] = None


AUTH = {"Authorization": "Bearer dev-token-123"}


def test_health_loading_then_ok(client):
    import app as app_module
    app_module.state["engine"], saved = None, app_module.state["engine"]
    r = client.get("/health")
    assert r.status_code == 503 and r.json() == {"status": "loading"}
    app_module.state["engine"] = saved
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"},
                                     {"Authorization": "bearer dev-token-123"},
                                     {"Authorization": "dev-token-123"}])
def test_auth_rejected(client, headers):
    for r in (client.get("/info", headers=headers),
              client.post("/predict", headers=headers, json={"id": "a", "context": "b", "question": "c"})):
        assert r.status_code == 401 and r.json() == {"error": "unauthorized"}


def test_info(client):
    r = client.get("/info", headers=AUTH)
    assert r.status_code == 200
    assert {"team", "model", "params"} <= r.json().keys()


@pytest.mark.parametrize("body", [
    {"id": "a", "context": "b"},                                  # missing field
    {"id": "", "context": "b", "question": "c"},                  # empty id
    {"id": "a" * 65, "context": "b", "question": "c"},            # id too long
    {"id": "a", "context": "b" * 5001, "question": "c"},          # context too long
    {"id": 7, "context": "b", "question": "c"},                   # wrong type
    {"id": "a", "context": "b", "question": "c", "extra": 1},     # extra field
    [1, 2, 3],                                                    # not an object
])
def test_bad_request(client, body):
    r = client.post("/predict", headers=AUTH, json=body)
    assert r.status_code in (400, 422) and r.json()["error"] == "bad_request"


def test_invalid_json(client):
    r = client.post("/predict", headers={**AUTH, "Content-Type": "application/json"}, content=b"{not json")
    assert r.status_code == 400 and r.json()["error"] == "bad_request"


def test_predict_answer_and_null(client, engine):
    it = SAMPLES[1]
    a = it["answers"][0]
    engine.target = (a["start"], a["start"] + len(a["text"]))
    body = {k: it[k] for k in ("id", "context", "question")}
    r = client.post("/predict", headers=AUTH, json=body).json()
    assert r["id"] == it["id"] and r["answer"] == a["text"] and r["start"] == a["start"]
    engine.target = None
    r = client.post("/predict", headers=AUTH, json=body).json()
    assert r["answer"] is None and r["start"] is None and 0 <= r["score"] <= 1
