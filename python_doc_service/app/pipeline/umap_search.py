import numpy as np
import umap
from app.pipeline.logger import get_logger
log = get_logger("umap")


def get_umap_candidates(n_chunks: int):
    if n_chunks < 5:
        # Not enough data to cluster meaningfully
        return [], []

    max_components = min(15, n_chunks - 2)

    n_components_candidates = list(
        set([
            5,
            8,
            10,
            int(np.log2(n_chunks))
        ])
    )

    n_components_candidates = [
        n for n in n_components_candidates
        if 2 <= n <= max_components
    ]

    n_neighbors_candidates = sorted(
        set([
            max(2, n_chunks // 10),
            max(5, n_chunks // 5),
            max(10, n_chunks // 2)
        ])
    )
    log.info(
    f"UMAP candidates | "
    f"n_components={n_components_candidates}, "
    f"n_neighbors={n_neighbors_candidates}"
)


    return n_components_candidates, n_neighbors_candidates



def run_umap(embeddings, n_neighbors, n_components):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42
    )
    return reducer.fit_transform(embeddings)
