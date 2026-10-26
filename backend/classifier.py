import json
import threading
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "model"
MAX_LEN = 256


class Classifier:
    def __init__(self):
        cfg = json.load(open(MODEL_DIR / "final_config.json"))
        self.labels = cfg["labels"]
        self.thresholds = np.array(cfg["scibert_only"]["thresholds"], dtype=np.float32)
        self.force_top1 = cfg["force_top1"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = MODEL_DIR / "scibert_final"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(path).to(self.device).eval())
        self._lock = threading.Lock()

    @staticmethod
    def build_text(title, abstract):
        abstract = " ".join(abstract.split())
        title = " ".join(title.split()) if title else ""
        return f"{title}. {abstract}" if title else abstract

    @torch.inference_mode()
    def predict_proba(self, title, abstract):
        text = self.build_text(title, abstract)
        enc = self.tok(text, truncation=True, max_length=MAX_LEN,
                       return_tensors="pt").to(self.device)
        with self._lock:
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(**enc).logits[0]
            else:
                logits = self.model(**enc).logits[0]
        return torch.sigmoid(logits.float()).cpu().numpy()

    def predict(self, title, abstract, top_k=10):
        probs = self.predict_proba(title, abstract)
        decided = probs >= self.thresholds
        if self.force_top1:
            decided[probs.argmax()] = True

        order = np.argsort(-probs)
        shown = [j for j in order if decided[j]]
        shown += [j for j in order if not decided[j]][: max(0, top_k - len(shown))]
        return [{"label": self.labels[j],
                 "score": round(float(probs[j]), 4),
                 "predicted": bool(decided[j])} for j in shown]