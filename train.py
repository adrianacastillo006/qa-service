"""
train.py - Fine-tune a BASE encoder on SQuAD 2.0 for extractive QA,
then tune the no-answer threshold on validation data.

Example (GPU):
  python train.py --base_model distilbert-base-uncased --epochs 2 --batch_size 16

Quick smoke test (CPU, minutes):
  python train.py --train_samples 2000 --val_samples 300 --epochs 1

Outputs
  model/                 fine-tuned weights + tokenizer + qa_meta.json (NOT committed to git)
  logs/train_log.json    loss history (training + validation)
  logs/loss.png          loss curve
  logs/threshold.json    threshold sweep + validation EM/F1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator, set_seed)

from metrics import item_scores
from qa_engine import QAEngine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="distilbert-base-uncased",
                   help="BASE checkpoint (never one already fine-tuned on SQuAD)")
    p.add_argument("--output_dir", default="model")
    p.add_argument("--train_samples", type=int, default=0, help="0 = full train set (~130k)")
    p.add_argument("--val_samples", type=int, default=2000, help="examples used for threshold tuning")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ------------------------------------------------------------ preprocessing
def make_train_features(tokenizer, max_len, stride):
    """Turn (question, context, answer) into token windows with start/end token labels.
    If the answer is not inside a window (or the question is unanswerable),
    the label points to the [CLS] token = 'no answer'."""
    def fn(examples):
        questions = [q.strip() for q in examples["question"]]
        enc = tokenizer(questions, examples["context"], truncation="only_second",
                        max_length=max_len, stride=stride, return_overflowing_tokens=True,
                        return_offsets_mapping=True, padding="max_length")
        sample_map = enc.pop("overflow_to_sample_mapping")
        offsets_all = enc.pop("offset_mapping")
        starts, ends = [], []
        for i, offsets in enumerate(offsets_all):
            cls_index = enc["input_ids"][i].index(tokenizer.cls_token_id)
            seq_ids = enc.sequence_ids(i)
            answers = examples["answers"][sample_map[i]]
            if len(answers["answer_start"]) == 0:           # unanswerable
                starts.append(cls_index); ends.append(cls_index); continue
            s_char = answers["answer_start"][0]
            e_char = s_char + len(answers["text"][0])
            ctx_start = seq_ids.index(1)
            ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)
            if not (offsets[ctx_start][0] <= s_char and offsets[ctx_end][1] >= e_char):
                starts.append(cls_index); ends.append(cls_index); continue  # answer not in this window
            ts = ctx_start
            while ts <= ctx_end and offsets[ts][0] <= s_char:
                ts += 1
            te = ctx_end
            while te >= ctx_start and offsets[te][1] >= e_char:
                te -= 1
            starts.append(ts - 1); ends.append(te + 1)
        enc["start_positions"] = starts
        enc["end_positions"] = ends
        return enc
    return fn


# ------------------------------------------------------- threshold tuning
def tune_threshold(engine: QAEngine, val_examples) -> dict:
    """For every validation example compute diff = null_score - best_span_score once,
    then sweep the threshold: predict null iff diff > threshold. Keep the best F1."""
    rows = []
    for ex in val_examples:
        rs = engine.raw_scores(ex["question"], ex["context"])
        golds = ex["answers"]["text"]
        diff = rs.null_score - rs.best_score
        em_ans, f1_ans = item_scores(rs.best_text, golds)   # if we answer
        em_null, f1_null = item_scores(None, golds)          # if we abstain
        rows.append((diff, em_ans, f1_ans, em_null, f1_null, bool(golds)))

    diffs = sorted({r[0] for r in rows})
    candidates = [-1e9] + [d + 1e-6 for d in diffs]  # thresholds between observed values
    best = None
    for t in candidates:
        em = np.mean([r[3] if r[0] > t else r[1] for r in rows])
        f1 = np.mean([r[4] if r[0] > t else r[2] for r in rows])
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(t), "em": float(em), "f1": float(f1)}

    def report(t):
        has = [r for r in rows if r[5]]
        no = [r for r in rows if not r[5]]
        pick = lambda r, i_null, i_ans: r[i_null] if r[0] > t else r[i_ans]
        return {
            "threshold": t,
            "em_all": float(np.mean([pick(r, 3, 1) for r in rows])),
            "f1_all": float(np.mean([pick(r, 4, 2) for r in rows])),
            "em_answerable": float(np.mean([pick(r, 3, 1) for r in has])) if has else None,
            "f1_answerable": float(np.mean([pick(r, 4, 2) for r in has])) if has else None,
            "acc_unanswerable": float(np.mean([pick(r, 3, 1) for r in no])) if no else None,
        }

    return {"n_examples": len(rows), "best": report(best["threshold"]), "zero_threshold": report(0.0),
            "sweep": [report(t) for t in np.linspace(-6, 6, 25).tolist()]}


# ---------------------------------------------------------------- plotting
def plot_loss(history, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = [(h["step"], h["loss"]) for h in history if "loss" in h]
    ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    plt.figure(figsize=(7, 4))
    if tr:
        plt.plot(*zip(*tr), label="train loss")
    if ev:
        plt.plot(*zip(*ev), "o-", label="validation loss")
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Fine-tuning on SQuAD 2.0")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(out_png, dpi=120)


# -------------------------------------------------------------------- main
def main():
    args = parse_args()
    set_seed(args.seed)
    Path("logs").mkdir(exist_ok=True)
    t_start = time.time()

    raw = load_dataset("rajpurkar/squad_v2")
    train_raw, val_raw = raw["train"], raw["validation"]
    if args.train_samples:
        train_raw = train_raw.shuffle(seed=args.seed).select(range(args.train_samples))
    val_raw = val_raw.shuffle(seed=args.seed)
    val_for_threshold = val_raw.select(range(min(args.val_samples, len(val_raw))))
    val_for_loss = val_raw.select(range(len(val_for_threshold), min(len(val_for_threshold) + 1000, len(val_raw))))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)
    fn = make_train_features(tokenizer, args.max_seq_length, args.doc_stride)
    train_ds = train_raw.map(fn, batched=True, remove_columns=train_raw.column_names)
    eval_ds = val_for_loss.map(fn, batched=True, remove_columns=val_for_loss.column_names)
    print(f"train examples={len(train_raw)} -> features={len(train_ds)}")

    targs = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_ratio,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=50,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=default_data_collator)
    trainer.train()
    trainer.evaluate()

    out = Path(args.output_dir)
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    history = trainer.state.log_history
    Path("logs/train_log.json").write_text(json.dumps(history, indent=1))
    plot_loss(history, "logs/loss.png")

    # Threshold tuning with the SAME engine the API uses
    meta = {"base_model": args.base_model, "max_seq_length": args.max_seq_length,
            "doc_stride": args.doc_stride, "null_threshold": 0.0}
    (out / "qa_meta.json").write_text(json.dumps(meta, indent=1))
    engine = QAEngine(out)
    result = tune_threshold(engine, val_for_threshold)
    meta["null_threshold"] = result["best"]["threshold"]
    (out / "qa_meta.json").write_text(json.dumps(meta, indent=1))

    result["training"] = {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()}
    result["training"]["train_features"] = len(train_ds)
    result["training"]["minutes"] = round((time.time() - t_start) / 60, 1)
    result["training"]["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    Path("logs/threshold.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["best"], indent=1))
    print(f"Saved model to {out}/ (threshold={meta['null_threshold']:.3f})")


if __name__ == "__main__":
    main()
