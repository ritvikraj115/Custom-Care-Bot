import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def inspect_neighbors(chunks, embeddings, idx, k=5):
    """
    Inspect nearest neighbors of a single chunk embedding.
    """

    sims = cosine_similarity(
        embeddings[idx:idx+1],
        embeddings
    )[0]

    top = sims.argsort()[::-1][1:k+1]

    print("\n================ VECTOR QUALITY ================\n")
    print("QUERY CHUNK:\n")
    print(chunks[idx]["text"][:300])
    print("\nNEAREST NEIGHBORS:\n")

    for i in top:
        print(f"[sim={sims[i]:.3f}]")
        print(chunks[i]["text"][:300])
        print("-" * 60)
