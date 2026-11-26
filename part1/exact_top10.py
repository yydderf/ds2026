from sentence_transformers import SentenceTransformer
from pprint import pprint

from common import load_embed

import pandas as pd, numpy as np

p = pd.read_csv("WANDS/dataset/product.csv", sep="\t").fillna("")
X, q = load_embed()

eligible = np.flatnonzero(p.product_class.to_numpy() == "Massage Chairs")

scores = X[eligible] @ q
order = np.argsort(-scores)[:10]
top_ids = eligible[order]
pprint([(p.iloc[i].product_name, float(s))
       for i, s in zip(top_ids, scores[order])])
