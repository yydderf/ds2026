from sentence_transformers import SentenceTransformer
from pprint import pprint
import pandas as pd, numpy as np

p = pd.read_csv("WANDS/dataset/product.csv", sep="\t").fillna("")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = (p.product_name + " " + p.product_description).tolist()
X = model.encode(texts, normalize_embeddings=True).astype("float32")

q = model.encode("salon chair", normalize_embeddings=True)
eligible = np.flatnonzero(p.product_class.to_numpy() == "Massage Chairs")

scores = X[eligible] @ q
order = np.argsort(-scores)[:10]
top_ids = eligible[order]
pprint([(p.iloc[i].product_name, float(s))
       for i, s in zip(top_ids, scores[order])])
