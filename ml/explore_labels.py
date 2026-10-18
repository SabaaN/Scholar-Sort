import pandas as pd
from collections import Counter

df = pd.read_parquet("data/arxiv_clean.parquet")

counts = Counter(c for cats in df["categories"] for c in cats)
print("distinct categories:", len(counts))
print()
for c, n in counts.most_common(90):
    print(f"{c:20s} {n:8d}")