# import torch
# print(torch.cuda.is_available(), torch.cuda.get_device_name(0))


# import pandas as pd
# df = pd.read_parquet("data/arxiv_clean.parquet")
# print(len(df))
# print(df["year"].value_counts().sort_index().tail(10))
# print(df["n_words"].describe())
# print(df["categories"].str.len().value_counts().head())   # labels per paper
# print(df.sample(3, random_state=0)[["title", "categories"]])

import pandas as pd
test = pd.read_parquet("data/test.parquet")
print(test["labels"].explode().value_counts().get("astro-ph", 0))