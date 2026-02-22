import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def run_hdbscan(reduced_embeddings, min_cluster_size, min_samples):
    # PRECOMPUTED cosine distance (EXACTLY like Colab)
    distance_matrix = 1 - cosine_similarity(reduced_embeddings)

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    return clusterer.fit_predict(distance_matrix)
