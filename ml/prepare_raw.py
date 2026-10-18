import json
import pandas as pd
from tqdm import tqdm

SRC = "data/arxiv-metadata-oai-snapshot.json"
OUT = "data/arxiv_clean.parquet"

rows = []
with open(SRC, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="reading"):
        d = json.loads(line)

        title = " ".join(d["title"].split())
        abstract = " ".join(d["abstract"].split())
        cats = d["categories"].split()

        # date of first submission (used for the time-based split)
        date = d["versions"][0]["created"] if d.get("versions") else d["update_date"]

        rows.append({
            "id": d["id"],
            "title": title,
            "abstract": abstract,
            "categories": cats,
            "date": date,
        })

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True, errors="coerce")
df = df.dropna(subset=["date"])
df["year"] = df["date"].dt.year
df["n_words"] = df["abstract"].str.split().str.len()

# basic cleaning
df = df[(df["n_words"] >= 30) & (df["n_words"] <= 500)]
df = df.drop_duplicates(subset="abstract")

df.to_parquet(OUT, index=False)
print(df.shape)
print(df.head())