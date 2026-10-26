import numpy as np
import pandas as pd
from classifier import Classifier

clf = Classifier()
test = pd.read_parquet("../data/test.parquet").iloc[:200]
saved = np.load("../ml/artifacts/scibert_test.npy")[:200]

mine = np.stack([clf.predict_proba(t, a) for t, a in zip(test["title"], test["abstract"])])

diff = np.abs(mine - saved)
print("max abs diff:", diff.max())
print("mean abs diff:", diff.mean())
print("top-1 agreement:", (mine.argmax(1) == saved.argmax(1)).mean())