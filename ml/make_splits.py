import json
import numpy as np
import pandas as pd
from collections import Counter

MIN_COUNT = 5000      # min train-period papers for a label to be kept
MAX_LABELS = 60
N_TRAIN, N_VAL, N_TEST = 100_000, 20_000, 30_000
SEED = 42

ALIAS = {
    "cs.IT": "math.IT", "cs.SY": "eess.SY", "math.MP": "math-ph",
    "stat.TH": "math.ST", "cs.NA": "math.NA",
}
# note: math.IT, eess.SY etc. are the canonical names kept as targets above;
# if you prefer the cs.* names, flip the mapping. Either way works.

df = pd.read_parquet("data/arxiv_clean.parquet")
df["categories"] = df["categories"].apply(
    lambda cs: sorted({ALIAS.get(c, c) for c in cs})
)

utc = lambda s: pd.Timestamp(s, tz="UTC")
train_pool = df[df["date"] < utc("2025-01-01")]
val_pool = df[(df["date"] >= utc("2025-01-01")) & (df["date"] < utc("2025-07-01"))]
test_pool = df[df["date"] >= utc("2025-07-01")]
print("pools:", len(train_pool), len(val_pool), len(test_pool))

# label set from the train period only
train_counts = Counter(c for cats in train_pool["categories"] for c in cats)
DROP = {"astro-ph"}   # legacy label, not used in recent papers
labels = [c for c, n in train_counts.most_common(MAX_LABELS + 5)
          if n >= MIN_COUNT and c not in DROP][:MAX_LABELS]
label_set = set(labels)
print(f"kept {len(labels)} labels")

def restrict(pool):
    pool = pool.copy()
    pool["labels"] = pool["categories"].apply(lambda cs: [c for c in cs if c in label_set])
    return pool[pool["labels"].str.len() > 0]

train_pool, val_pool, test_pool = map(restrict, (train_pool, val_pool, test_pool))

# rebalanced train sample
freq = Counter(c for ls in train_pool["labels"] for c in ls)
w = train_pool["labels"].apply(lambda ls: 1.0 / np.sqrt(min(freq[l] for l in ls)))
train = train_pool.sample(N_TRAIN, weights=w, random_state=SEED)

# natural-distribution val/test
val = val_pool.sample(min(N_VAL, len(val_pool)), random_state=SEED)
test = test_pool.sample(min(N_TEST, len(test_pool)), random_state=SEED)

cols = ["id", "title", "abstract", "labels", "date"]
train[cols].to_parquet("data/train.parquet", index=False)
val[cols].to_parquet("data/val.parquet", index=False)
test[cols].to_parquet("data/test.parquet", index=False)
json.dump(labels, open("data/labels.json", "w"), indent=2)

# report
for name, d in [("train", train), ("val", val), ("test", test)]:
    c = Counter(l for ls in d["labels"] for l in ls)
    print(f"\n{name}: {len(d)} rows, {d['labels'].str.len().mean():.2f} labels/paper")
    print("  rarest 8 labels:", sorted(c.items(), key=lambda x: x[1])[:8])
    print("  most common 5:  ", c.most_common(5))