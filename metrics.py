"""
metrics.py - Official-style SQuAD 2.0 metrics (Exact Match and F1).
No heavy dependencies, so it can be used by train.py and by the check scripts.
"""
import collections
import re
import string


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def item_scores(pred, golds):
    """(EM, F1) of one prediction. pred=None means 'no answer'."""
    if not golds:                                   # unanswerable item
        return (1.0, 1.0) if pred is None else (0.0, 0.0)
    if pred is None:
        return 0.0, 0.0
    em = max(float(normalize(pred) == normalize(g)) for g in golds)
    f1 = max(f1_score(pred, g) for g in golds)
    return em, f1


