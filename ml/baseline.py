import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from eval_utils import evaluate, best_global_threshold, per_label_f1

ART = Path("ml/artifacts")
ART.mkdir(parents=True, exist_ok=True)

# ---------- load ----------
labels = json.load(open("data/labels.json"))
train = pd.read_parquet("data/train.parquet")
val = pd.read_parquet("data/val.parquet")
test = pd.read_parquet("data/test.parquet")


def make_text(df):
    return (df["title"] + ". " + df["abstract"]).tolist()


mlb = MultiLabelBinarizer(classes=labels)
Y_train = mlb.fit_transform(train["labels"])
Y_val = mlb.transform(val["labels"])
Y_test = mlb.transform(test["labels"])

# ---------- features ----------
t0 = time.time()
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.5,
    max_features=400_000,
    sublinear_tf=True,
    strip_accents="unicode",
    stop_words="english",
)
X_train = vec.fit_transform(make_text(train))     # fit on train ONLY
X_val = vec.transform(make_text(val))
X_test = vec.transform(make_text(test))
print(f"features: {X_train.shape}  ({time.time() - t0:.0f}s)")

# ---------- model ----------
t0 = time.time()
clf = OneVsRestClassifier(
    LogisticRegression(C=10, solver="liblinear"),
    n_jobs=-1,
)
clf.fit(X_train, Y_train)
print(f"trained in {time.time() - t0:.0f}s")

val_prob = clf.predict_proba(X_val)
test_prob = clf.predict_proba(X_test)

# ---------- evaluate ----------
thr, _ = best_global_threshold(Y_val, val_prob)
print(f"\nbest global threshold on val: {thr:.3f}")

def show(name, y, p, t):
    r = evaluate(y, p, t)
    print(f"{name:28s}" + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))

show("val  @0.5", Y_val, val_prob, 0.5)
show(f"val  @{thr:.2f} (tuned)", Y_val, val_prob, thr)
show("test @0.5", Y_test, test_prob, 0.5)
show(f"test @{thr:.2f} (val-tuned)", Y_test, test_prob, thr)

print("\nWeakest 10 labels on test:")
for lab, f1, n in per_label_f1(Y_test, test_prob, labels, thr)[:10]:
    print(f"  {lab:22s} f1={f1:.3f}  support={n}")

# ---------- save ----------
joblib.dump({"vec": vec, "clf": clf, "labels": labels, "threshold": thr},
            ART / "baseline.joblib")
np.save(ART / "baseline_val.npy", val_prob)
np.save(ART / "baseline_test.npy", test_prob)