import numpy as np

from app.pipeline.umap_search import get_umap_candidates, run_umap
from app.pipeline.hdbscan_runner import run_hdbscan
from app.pipeline.scoring import clustering_score
from app.pipeline.logger import get_logger
log = get_logger("clustering")


def find_best_clustering(embeddings):
    n_chunks = len(embeddings)

    n_components_candidates, n_neighbors_candidates = get_umap_candidates(n_chunks)

    results = []
    log.info(f"Finding best clustering for {n_chunks} chunks")
    log.debug(
    f"Grid size: "
    f"{len(n_components_candidates)} × "
    f"{len(n_neighbors_candidates)} × 3 × 2"
)

    for nc in n_components_candidates:
        for nn in n_neighbors_candidates:
            # EXACT cast you did in Colab
            reduced = run_umap(
                embeddings,
                n_neighbors=nn,
                n_components=nc
            ).astype(np.float64)
            min_cluster_size = max(3, n_chunks // 8)


            for mcs in [min_cluster_size,5, 10, 20]:
                for ms in [1, 5]:
                    labels = run_hdbscan(reduced, mcs, ms)
                    score = clustering_score(embeddings, labels)

                    results.append({
                        "n_components": nc,
                        "n_neighbors": nn,
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "score": score,
                        "clusters": len(set(labels)) - (1 if -1 in labels else 0),
                        "noise": float(np.mean(labels == -1))
                    })

    # BEST CONFIG (exact logic)
    best = sorted(results, key=lambda x: x["score"], reverse=True)[0]

    # RE-RUN with best params (mandatory)
    best_reduced = run_umap(
        embeddings,
        best["n_neighbors"],
        best["n_components"]
    ).astype(np.float64)

    best_labels = run_hdbscan(
        best_reduced,
        best["min_cluster_size"],
        best["min_samples"]
    )
    log.info(f"Total configs evaluated: {len(results)}")
    log.info(
    f"Best params: {best}"
)
    log.info(
    f"Clusters formed: "
    f"{len(set(labels)) - (1 if -1 in labels else 0)}, "
    f"Noise ratio: {float((labels == -1).mean()):.2f}"
)
    confidence = 1 - best["noise"]
    log.info(f"Cluster confidence: {confidence:.2f}")




    return best_labels, best, results
