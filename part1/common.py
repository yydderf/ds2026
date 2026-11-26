from pathlib import Path

import subprocess, sys
import numpy as np

curr_dir = Path(__file__).resolve().parent

def load_embed():
    X_path, q_path = curr_dir / "X.npy", curr_dir / "q.npy"
    if not (X_path.exists() and q_path.exists()):
        print("embeddings not found, running embed.py")
        subprocess.run([sys.executable, str(curr_dir / "embed.py")], cwd=curr_dir, check=True)
    return np.load(X_path), np.load(q_path)
