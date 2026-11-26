from pprint import pprint

from common import load_embed

import pandas as pd
import numpy as np
import faiss


# load data from csv
p = pd.read_csv("WANDS/dataset/product.csv", sep="\t").fillna("")

X, q = load_embed()

d = X.shape[1] # X, p, q from slide 6
coarse = faiss.IndexFlatIP(d)
ivf = faiss.IndexIVFFlat(coarse, d, 128, faiss.METRIC_INNER_PRODUCT)

ivf.train(X) # learn 128 cluster centroids
ivf.add(X)

params = faiss.SearchParametersIVF(nprobe=16)
scores, ids = ivf.search(q.reshape(1, -1), 100, params=params)
valid = [(int(i), float(s)) for i, s in zip(ids[0], scores[0])
         if i >= 0 and p.iloc[i].product_class == "Massage Chairs"]
pprint([(p.iloc[i].product_name, s) for i, s in valid[:10]])
