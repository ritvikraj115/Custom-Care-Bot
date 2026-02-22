import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def clustering_score(embeddings, labels):
    clusters = {}

    for i, lbl in enumerate(labels):
        if lbl == -1:
            continue
        clusters.setdefault(lbl, []).append(i)

    # Reject degenerate solutions
    if len(clusters) < 2:
        return -1

    coherences = []
    centroids = []

    for idxs in clusters.values():
        vecs = embeddings[idxs]
        centroid = vecs.mean(axis=0)
        centroids.append(centroid)
        coherences.append(np.mean(cosine_similarity(vecs, vecs)))

    coherence = np.mean(coherences)
    centroid_similarity = np.mean(cosine_similarity(centroids, centroids))
    noise_ratio = np.mean(labels == -1)

    score = (
        0.45 * coherence
        - 0.35 * centroid_similarity
        - 0.20 * noise_ratio
    )

    return float(score)
