import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def inspect_cluster(chunks, cluster_id, n=5):
    items = [c for c in chunks if c["cluster"] == cluster_id]

    print(f"\n================ CLUSTER {cluster_id} ================\n")
    print(f"Total chunks: {len(items)}\n")

    for c in items[:n]:
        print(c["text"][:300])
        print("-" * 60)


def cluster_coherence(chunks, embeddings, cluster_id):
    idxs = [
        i for i, c in enumerate(chunks)
        if c["cluster"] == cluster_id
    ]

    if len(idxs) < 2:
        return None

    sims = cosine_similarity(
        embeddings[idxs],
        embeddings[idxs]
    )
    return float(np.mean(sims))


def centroid_similarity(chunks, embeddings):
    clusters = defaultdict(list)

    for i, c in enumerate(chunks):
        if c["cluster"] != -1:
            clusters[c["cluster"]].append(embeddings[i])

    centroids = [
        np.mean(v, axis=0)
        for v in clusters.values()
    ]

    return cosine_similarity(centroids, centroids)
