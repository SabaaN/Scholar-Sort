import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

MODEL = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-base-en-v1.5"
TAG = MODEL.split("/")[-1]
MAX_LEN, BS = 256, 128
ART = Path("ml/artifacts")
ART.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).cuda().eval()


@torch.no_grad()
def embed(texts):
    order = np.argsort([len(t) for t in texts])          # sort by length -> less padding
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in tqdm(range(0, len(texts), BS), desc="embedding"):
        idx = order[i:i + BS]
        batch = tok([texts[j] for j in idx], padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            h = model(**batch).last_hidden_state[:, 0]   # [CLS] token vector
        h = torch.nn.functional.normalize(h.float(), dim=-1)
        out[idx] = h.cpu().numpy()                       # put back in original order
    return out


for split in ["train", "val", "test"]:
    df = pd.read_parquet(f"data/{split}.parquet")
    texts = (df["title"] + ". " + df["abstract"]).tolist()
    np.save(ART / f"emb_{TAG}_{split}.npy", embed(texts))
    print(split, "done")