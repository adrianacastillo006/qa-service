"""
qa_engine.py - Extractive QA inference (SQuAD 2.0 style).

This module is used in TWO places, so the logic is identical in both:
  1. train.py  -> to tune the no-answer threshold on validation data.
  2. app.py    -> to answer /predict requests.

Pipeline for one (question, context) pair:
  tokenize into overlapping windows -> run the encoder -> for each window,
  find the best valid span (start_logit + end_logit) and the "no answer"
  score ([CLS] start + [CLS] end) -> compare them with the threshold ->
  map token offsets back to characters of the ORIGINAL context.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MAX_SEQ_LENGTH = 384   # tokens per window (question + context + special tokens)
DEFAULT_DOC_STRIDE = 128       # tokens of overlap between consecutive windows
DEFAULT_MAX_ANSWER_TOKENS = 30 # longest answer span we allow, in tokens
DEFAULT_N_BEST = 20            # candidate start/end positions inspected per window
MAX_QUESTION_TOKENS = 128      # very long questions are cut so a window always fits


@dataclass
class RawScores:
    """Everything needed to decide 'span' vs 'null' for one example."""
    best_text: str | None   # best span text (None if no valid span exists)
    best_start: int | None  # its character offset in the context
    best_score: float       # start_logit + end_logit of the best span
    null_score: float       # start_logit[CLS] + end_logit[CLS] (min over windows)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))


class QAEngine:
    def __init__(self, model_dir: str | os.PathLike,
                 max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
                 doc_stride: int = DEFAULT_DOC_STRIDE,
                 threshold: float | None = None,
                 device: str | None = None):
        # torch / transformers are imported here so unit tests can run without them.
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_dir)
        self.model.to(self.device).eval()

        meta = self._read_meta()
        self.base_model = meta.get("base_model", "unknown")
        self.max_seq_length = int(meta.get("max_seq_length", max_seq_length))
        self.doc_stride = int(meta.get("doc_stride", doc_stride))
        # Priority: explicit argument > env var > value saved by train.py > 0.0
        if threshold is None:
            env_t = os.environ.get("QA_NULL_THRESHOLD")
            threshold = float(env_t) if env_t else float(meta.get("null_threshold", 0.0))
        self.threshold = threshold
        self.n_params = sum(p.numel() for p in self.model.parameters())

    # ------------------------------------------------------------------ setup
    def _read_meta(self) -> dict:
        path = self.model_dir / "qa_meta.json"
        return json.loads(path.read_text()) if path.exists() else {}

    # ----------------------------------------------------------- tokenization
    def _shorten_question(self, question: str) -> str:
        """Cut questions longer than MAX_QUESTION_TOKENS so windows never overflow."""
        enc = self.tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= MAX_QUESTION_TOKENS:
            return question
        return question[: enc["offset_mapping"][MAX_QUESTION_TOKENS - 1][1]]

    def _encode(self, question: str, context: str):
        """Split the context into overlapping windows of max_seq_length tokens."""
        return self.tokenizer(
            self._shorten_question(question.strip()),
            context,
            truncation="only_second",          # only the context is cut into windows
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,     # -> several windows for long contexts
            return_offsets_mapping=True,        # token -> (char_start, char_end)
            padding="longest",
        )

    # ---------------------------------------------------------------- forward
    def _logits(self, enc) -> tuple[np.ndarray, np.ndarray]:
        """Run the encoder on all windows at once. Returns (start, end) logits [n_windows, seq_len]."""
        torch = self.torch
        inputs = {k: torch.tensor(enc[k], device=self.device)
                  for k in self.tokenizer.model_input_names if k in enc}
        with torch.inference_mode():
            out = self.model(**inputs)
        return out.start_logits.float().cpu().numpy(), out.end_logits.float().cpu().numpy()

    # ---------------------------------------------------------- post-process
    def raw_scores(self, question: str, context: str,
                   max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS,
                   n_best: int = DEFAULT_N_BEST) -> RawScores:
        enc = self._encode(question, context)
        start_logits, end_logits = self._logits(enc)

        best_score, best_span = -1e9, None
        null_score = 1e9
        for w in range(len(enc["input_ids"])):
            seq_ids = enc.sequence_ids(w)           # None=special, 0=question, 1=context
            offsets = enc["offset_mapping"][w]
            s_log, e_log = start_logits[w], end_logits[w]

            cls_idx = enc["input_ids"][w].index(self.tokenizer.cls_token_id)
            null_score = min(null_score, float(s_log[cls_idx] + e_log[cls_idx]))

            top_s = np.argsort(s_log)[::-1][:n_best]
            top_e = np.argsort(e_log)[::-1][:n_best]
            for s in top_s:
                if s >= len(seq_ids) or seq_ids[s] != 1:
                    continue
                for e in top_e:
                    if e >= len(seq_ids) or seq_ids[e] != 1:
                        continue
                    if e < s or e - s + 1 > max_answer_tokens:
                        continue
                    score = float(s_log[s] + e_log[e])
                    if score > best_score:
                        best_score = score
                        best_span = (offsets[s][0], offsets[e][1])

        if best_span is None:
            return RawScores(None, None, best_score, null_score)

        start_char, end_char = best_span
        text = context[start_char:end_char]
        # Trim surrounding whitespace but keep context[start:start+len] == answer
        stripped = text.lstrip()
        start_char += len(text) - len(stripped)
        text = stripped.rstrip()
        if not text:
            return RawScores(None, None, -1e9, null_score)
        return RawScores(text, start_char, best_score, null_score)

    def decide(self, rs: RawScores, threshold: float | None = None) -> dict:
        """Apply the no-answer rule: null if (null_score - best_score) > threshold."""
        t = self.threshold if threshold is None else threshold
        if rs.best_text is None:
            return {"answer": None, "start": None, "score": 1.0}
        margin = t - (rs.null_score - rs.best_score)   # >0 -> answer, <=0 -> null
        if margin > 0:
            return {"answer": rs.best_text, "start": rs.best_start, "score": round(_sigmoid(margin), 4)}
        return {"answer": None, "start": None, "score": round(_sigmoid(-margin), 4)}

    def predict(self, question: str, context: str) -> dict:
        return self.decide(self.raw_scores(question, context))
