import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer

from eval_utils import evaluate, best_global_threshold, per_label_f1

TAG = sys.argv[1] if len(sys.argv) > 1 else "bge-base-en-v1.5"
ART = Path("ml/artifacts")
torch.manual_seed(0)

labels = json.load(open("data/labels.json"))
mlb = MultiLabelBinarizer(classes=labels)
Y = {s: mlb.fit_transform(pd.read_parquet(f"data/{s}.parquet")["labels"]).astype(np.float32)
     for s in ["train", "val", "test"]}
X = {s: np.load(ART / f"emb_{TAG}_{s}.npy") for s in ["train", "val", "test"]}

dev = "cuda"
Xtr, Ytr = torch.tensor(X["train"]).to(dev), torch.tensor(Y["train"]).to(dev)
Xva = torch.tensor(X["val"]).to(dev)
Xte = torch.tensor(X["test"]).to(dev)

dim = Xtr.shape[1]
head = nn.Sequential(
    nn.Linear(dim, 1024), nn.GELU(), nn.Dropout(0.3),
    nn.Linear(1024, len(labels)),
).to(dev)

EPOCHS, BS = 40, 256
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.BCEWithLogitsLoss()


def predict(x):
    head.eval()
    with torch.no_grad():
        return torch.sigmoid(head(x)).cpu().numpy()


best_map, best_state = 0, None
for ep in range(EPOCHS):
    head.train()
    perm = torch.randperm(len(Xtr), device=dev)
    for i in range(0, len(perm), BS):
        idx = perm[i:i + BS]
        loss = loss_fn(head(Xtr[idx]), Ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    sched.step()

    val_prob = predict(Xva)
    m = average_precision_score(Y["val"], val_prob, average="macro")
    if m > best_map:
        best_map = m
        best_state = {k: v.clone() for k, v in head.state_dict().items()}
    if ep % 5 == 4:
        print(f"epoch {ep+1:2d}  loss={loss.item():.4f}  val mAP={m:.4f}")

head.load_state_dict(best_state)
val_prob, test_prob = predict(Xva), predict(Xte)

thr, _ = best_global_threshold(Y["val"], val_prob)
print(f"\nbest threshold: {thr:.3f}")


def show(name, y, p, t):
    r = evaluate(y, p, t)
    print(f"{name:28s}" + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))


show("val  (tuned)", Y["val"], val_prob, thr)
show("test (val-tuned)", Y["test"], test_prob, thr)

# ---- ensemble with the TF-IDF baseline ----
b_val, b_test = np.load(ART / "baseline_val.npy"), np.load(ART / "baseline_test.npy")
ens_val, ens_test = (val_prob + b_val) / 2, (test_prob + b_test) / 2
thr_e, _ = best_global_threshold(Y["val"], ens_val)
print(f"\nensemble threshold: {thr_e:.3f}")
show("val  ensemble", Y["val"], ens_val, thr_e)
show("test ensemble", Y["test"], ens_test, thr_e)

print("\nWeakest 10 labels (embeddings, test):")
for lab, f1, n in per_label_f1(Y["test"], test_prob, labels, thr)[:10]:
    print(f"  {lab:22s} f1={f1:.3f}  support={int(n)}")

np.save(ART / f"{TAG}_val.npy", val_prob)
np.save(ART / f"{TAG}_test.npy", test_prob)
torch.save(head.state_dict(), ART / f"{TAG}_head.pt")