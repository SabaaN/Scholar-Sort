import sys
import json
import time
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

from eval_utils import evaluate, best_global_threshold, per_label_f1

# ---------- config ----------
SMOKE = "--smoke" in sys.argv
MODEL = "allenai/scibert_scivocab_uncased"
MAX_LEN = 256
BS, ACCUM = 16, 2            # effective batch size = 32
EPOCHS = 3
LR = 3e-5
WARMUP = 0.06
TITLE_DROP = 0.2             # fraction of training examples shown WITHOUT the title
ART = Path("ml/artifacts")
ART.mkdir(parents=True, exist_ok=True)

random.seed(0)
torch.manual_seed(0)
dev = "cuda"

# ---------- data ----------
labels = json.load(open("data/labels.json"))
train = pd.read_parquet("data/train.parquet")
val = pd.read_parquet("data/val.parquet")
test = pd.read_parquet("data/test.parquet")
if SMOKE:
    train, val, test = train.iloc[:2000], val.iloc[:1000], test.iloc[:1000]
    EPOCHS = 1

mlb = MultiLabelBinarizer(classes=labels)
mlb.fit([labels])
Y = {n: mlb.transform(d["labels"]).astype(np.float32)
     for n, d in [("train", train), ("val", val), ("test", test)]}

tok = AutoTokenizer.from_pretrained(MODEL)


class PaperDS(Dataset):
    def __init__(self, df, y):
        self.t, self.a, self.y = df["title"].tolist(), df["abstract"].tolist(), y

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        return self.t[i], self.a[i], self.y[i]


def collate(batch, train_mode):
    texts = []
    for t, a, _ in batch:
        if train_mode and random.random() < TITLE_DROP:
            texts.append(a)                       # abstract only
        else:
            texts.append(t + ". " + a)
    enc = tok(texts, padding=True, truncation=True,
              max_length=MAX_LEN, return_tensors="pt")
    enc["labels"] = torch.tensor(np.stack([b[2] for b in batch]))
    return enc


train_loader = DataLoader(PaperDS(train, Y["train"]), batch_size=BS, shuffle=True,
                          collate_fn=partial(collate, train_mode=True))

# ---------- model ----------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(labels), problem_type="multi_label_classification"
).to(dev)

no_decay = ["bias", "LayerNorm.weight"]
groups = [
    {"params": [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
opt = torch.optim.AdamW(groups, lr=LR)
total_steps = (len(train_loader) // ACCUM) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(WARMUP * total_steps), total_steps)
loss_fn = nn.BCEWithLogitsLoss()


@torch.no_grad()
def predict(df, bs=64):
    model.eval()
    texts = (df["title"] + ". " + df["abstract"]).tolist()
    order = np.argsort([len(t) for t in texts])          # less padding
    out = np.zeros((len(texts), len(labels)), dtype=np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits
        out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


# ---------- train ----------
best_map, best_state = 0.0, None
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    run_loss = 0.0
    for step, batch in enumerate(train_loader):
        batch = batch.to(dev)
        y = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits
        loss = loss_fn(logits.float(), y) / ACCUM
        loss.backward()
        run_loss += loss.item() * ACCUM

        if (step + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        if (step + 1) % 200 == 0:
            print(f"ep {ep+1} step {step+1}/{len(train_loader)}  "
                  f"loss={run_loss/200:.4f}  {(time.time()-t0)/60:.1f} min", flush=True)
            run_loss = 0.0

    val_prob = predict(val)
    m = average_precision_score(Y["val"][:, Y["val"].sum(0) > 0],
                                val_prob[:, Y["val"].sum(0) > 0], average="macro")
    print(f"== epoch {ep+1} done: val mAP={m:.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    if m > best_map:
        best_map = m
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# ---------- final evaluation with the best epoch ----------
model.load_state_dict(best_state)
val_prob, test_prob = predict(val), predict(test)
thr, _ = best_global_threshold(Y["val"], val_prob)
print(f"\nbest threshold: {thr:.3f}")


def show(name, y, p, t):
    r = evaluate(y, p, t)
    print(f"{name:28s}" + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))


show("val  (tuned)", Y["val"], val_prob, thr)
show("test (val-tuned)", Y["test"], test_prob, thr)

print("\nWeakest 10 labels (SciBERT, test):")
for lab, f1, n in per_label_f1(Y["test"], test_prob, labels, thr)[:10]:
    print(f"  {lab:22s} f1={f1:.3f}  support={int(n)}")

# ---------- ensemble with earlier models ----------
if not SMOKE:
    parts_val = [val_prob, np.load(ART / "baseline_val.npy"),
                 np.load(ART / "bge-base-en-v1.5_val.npy")]
    parts_test = [test_prob, np.load(ART / "baseline_test.npy"),
                  np.load(ART / "bge-base-en-v1.5_test.npy")]
    ens_val, ens_test = np.mean(parts_val, axis=0), np.mean(parts_test, axis=0)
    thr_e, _ = best_global_threshold(Y["val"], ens_val)
    print(f"\nensemble (SciBERT + TF-IDF + BGE) threshold: {thr_e:.3f}")
    show("val  ensemble", Y["val"], ens_val, thr_e)
    show("test ensemble", Y["test"], ens_test, thr_e)

# ---------- save ----------
if not SMOKE:
    np.save(ART / "scibert_val.npy", val_prob)
    np.save(ART / "scibert_test.npy", test_prob)
    model.save_pretrained(ART / "scibert_final")
    tok.save_pretrained(ART / "scibert_final")
    json.dump({"threshold": thr}, open(ART / "scibert_final" / "threshold.json", "w"))
    print("saved model to", ART / "scibert_final")