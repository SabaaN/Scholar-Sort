import numpy as np
from sklearn.metrics import f1_score, average_precision_score


def precision_at_k(y_true, y_score, k):
    topk = np.argsort(-y_score, axis=1)[:, :k]          # indices of k highest scores per paper
    hits = np.take_along_axis(y_true, topk, axis=1)     # 1 if that predicted label is correct
    return hits.mean()


def best_global_threshold(y_true, y_score, grid=None):
    grid = np.arange(0.05, 0.95, 0.025) if grid is None else grid
    scores = [f1_score(y_true, (y_score >= t).astype(int),
                       average="micro", zero_division=0) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def evaluate(y_true, y_score, threshold=0.5):
    keep = y_true.sum(axis=0) > 0                       # ignore labels with no positives
    y_pred = (y_score >= threshold).astype(int)
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true[:, keep], y_pred[:, keep], average="macro", zero_division=0),
        "p@1": precision_at_k(y_true, y_score, 1),
        "p@3": precision_at_k(y_true, y_score, 3),
        "mAP": average_precision_score(y_true[:, keep], y_score[:, keep], average="macro"),
    }


def per_label_f1(y_true, y_score, labels, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = y_true.sum(axis=0)
    rows = sorted(zip(labels, f1s, support), key=lambda r: r[1])
    return rows                                         # worst label first