import numpy as np
data = np.load(r"C:\Users\Ahmed\Desktop\KRJAM\rag\index.npz")
print(data["embeddings"].shape)  # --> (N, d)
