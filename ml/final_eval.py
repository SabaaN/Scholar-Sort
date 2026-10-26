import json
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer

ART = Path("ml/artifacts")
GRID = np.arange(0.05, 0.9, 0.025)
labels = json.load(open("data/labels.json"))
val = pd.read_parquet("data/val.parquet")
test = pd.read_parquet("data/test.parquet")

mlb = MultiLabelBinarizer(classes=labels)
mlb.fit([labels])
Yv, Yt = mlb.transform(val["labels"]), mlb.transform(test["labels"])

NAMES = ["scibert", "baseline", "bge-base-en-v1.5"]
P = {n: (np.load(ART / f"{n}_val.npy"), np.load(ART / f"{n}_test.npy")) for n in NAMES}


# ---------- helpers ----------
def mAP(Y, p):
    keep = Y.sum(0) > 0
    return average_precision_score(Y[:, keep], p[:, keep], average="macro")


def metrics(Y, pred):
    keep = Y.sum(0) > 0
    return (f1_score(Y, pred, average="micro", zero_division=0),
            f1_score(Y[:, keep], pred[:, keep], average="macro", zero_division=0))


def decide(p, thr):
    """label is ON if prob >= threshold; the top-1 label is always ON."""
    pred = (p >= thr).astype(int)
    pred[np.arange(len(p)), p.argmax(1)] = 1
    return pred


def best_global(Y, p):
    return float(max(GRID, key=lambda t: metrics(Y, decide(p, t))[0]))


def best_per_label(Y, p):
    """best threshold for each label alone; NaN if too few positives"""
    min_support = 0.005 * len(Y)
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        if Y[:, j].sum() < min_support:
            continue
        f = [f1_score(Y[:, j], p[:, j] >= t, zero_division=0) for t in GRID]
        out[j] = GRID[int(np.argmax(f))]
    return out


def blend(best, tg, alpha):
    """shrink per-label thresholds toward the global one"""
    return np.where(np.isnan(best), tg, alpha * np.nan_to_num(best) + (1 - alpha) * tg)


# ---------- 1. standalone models ----------
print("=== standalone models (test, global threshold tuned on val, top-1 forced) ===")
for n in NAMES:
    tg = best_global(Yv, P[n][0])
    mi, ma = metrics(Yt, decide(P[n][1], tg))
    print(f"{n:20s} thr={tg:.3f}  micro={mi:.4f}  macro={ma:.4f}  mAP={mAP(Yt, P[n][1]):.4f}")

# ---------- 2. ensemble weights (chosen on val mAP) ----------
best_w, best_m = None, 0
for a in range(3, 11):                      # SciBERT weight 0.3 ... 1.0
    for b in range(0, 11 - a):              # TF-IDF weight
        c = 10 - a - b                      # BGE weight
        w = np.array([a, b, c]) / 10
        pv = sum(wi * P[n][0] for wi, n in zip(w, NAMES))
        m = mAP(Yv, pv)
        if m > best_m:
            best_m, best_w = m, w
print(f"\nbest ensemble weights (scibert, tfidf, bge): {best_w}  val mAP={best_m:.4f}")


# ---------- 3. thresholds: global vs per-label (checked with split-halves of val) ----------
def pipeline(name, pv, pt):
    print(f"\n=== {name} ===")
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(Yv))
    A, B = idx[: len(idx) // 2], idx[len(idx) // 2:]

    results = {}
    for alpha in [0.0, 0.5, 1.0]:
        scores = []
        for tr, ev in [(A, B), (B, A)]:     # tune on one half, test on the other
            tg = best_global(Yv[tr], pv[tr])
            thr = blend(best_per_label(Yv[tr], pv[tr]), tg, alpha)
            scores.append(metrics(Yv[ev], decide(pv[ev], thr)))
        results[alpha] = np.mean(scores, axis=0)
        print(f"  alpha={alpha:.1f}  held-out-half micro={results[alpha][0]:.4f} "
              f"macro={results[alpha][1]:.4f}")
    alpha = max(results, key=lambda a: results[a].sum())
    print(f"  -> chosen alpha = {alpha}  (0 = one global threshold, 1 = fully per-label)")

    tg = best_global(Yv, pv)
    thr = blend(best_per_label(Yv, pv), tg, alpha)
    mi0, ma0 = metrics(Yt, (pt >= 0.5).astype(int))
    mi1, ma1 = metrics(Yt, decide(pt, tg))
    mi2, ma2 = metrics(Yt, decide(pt, thr))
    print(f"  TEST fixed 0.5 threshold        micro={mi0:.4f} macro={ma0:.4f}")
    print(f"  TEST global {tg:.3f} + top-1       micro={mi1:.4f} macro={ma1:.4f}")
    print(f"  TEST final thresholds + top-1   micro={mi2:.4f} macro={ma2:.4f}")
    return tg, alpha, thr


sci_cfg = pipeline("SciBERT only", *P["scibert"])
pv_ens = sum(wi * P[n][0] for wi, n in zip(best_w, NAMES))
pt_ens = sum(wi * P[n][1] for wi, n in zip(best_w, NAMES))
ens_cfg = pipeline("Weighted ensemble", pv_ens, pt_ens)

# ---------- 4. error analysis (ensemble) ----------
pred = decide(pt_ens, ens_cfg[2])
f1s = f1_score(Yt, pred, average=None, zero_division=0)
weak = [j for j in np.argsort(f1s) if Yt[:, j].sum() >= 100][:6]

print("\n=== error analysis: 6 weakest labels ===")
for j in weak:
    fn = (Yt[:, j] == 1) & (pred[:, j] == 0)        # missed
    fp = (Yt[:, j] == 0) & (pred[:, j] == 1)        # wrongly predicted
    print(f"\n{labels[j]}  f1={f1s[j]:.3f}  support={Yt[:, j].sum()}  "
          f"missed={fn.sum()}  false alarms={fp.sum()}")

    # missed papers: what labels did they actually carry?
    carried = Counter(labels[k] for i in np.where(fn)[0] for k in np.where(Yt[i])[0] if k != j)
    print("  when missed, papers instead carry:", carried.most_common(4))

    # false alarms: what labels did those papers really have?
    really = Counter(labels[k] for i in np.where(fp)[0] for k in np.where(Yt[i])[0])
    print("  false alarms are really:", really.most_common(4))

    for i in np.where(fn)[0][:2]:
        print(f"  missed example: {test['title'].iloc[i][:90]!r}  true={test['labels'].iloc[i]}")

# ---------- 5. save configs ----------
json.dump({
    "labels": labels, "force_top1": True,
    "scibert_only": {"thresholds": sci_cfg[2].tolist()},
    "ensemble": {"weights": dict(zip(NAMES, best_w.tolist())),
                 "thresholds": ens_cfg[2].tolist()},
}, open(ART / "final_config.json", "w"), indent=2)
print("\nsaved ml/artifacts/final_config.json")