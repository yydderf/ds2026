from sentence_transformers import SentenceTransformer

import pandas as pd
import numpy as np

p = pd.read_csv("WANDS/dataset/product.csv", sep="\t").fillna("")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = (p.product_name + " " + p.product_description).tolist()

np.save("X.npy", model.encode(texts, normalize_embeddings=True).astype("float32"))
np.save("q.npy", model.encode("salon chair", normalize_embeddings=True).astype("float32"))

