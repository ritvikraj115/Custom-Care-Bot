import random

from app.pipeline.quality.vector_quality import inspect_neighbors
from app.pipeline.quality.cluster_quality import (
    inspect_cluster,
    cluster_coherence,
    centroid_similarity
)
from app.pipeline.quality.routing_quality import inspect_query_routing


def run_quality_checks(chunks, embeddings, client_id, bot_id):
    print("\n############################################")
    print("### RUNNING QUALITY CHECKS")
    print("############################################\n")

    # --------------------------------------------------
    # 1. VECTOR QUALITY (random chunk)
    # --------------------------------------------------
    idx = random.randint(0, len(chunks) - 1)
    inspect_neighbors(chunks, embeddings, idx)

    # --------------------------------------------------
    # 2. CLUSTER QUALITY
    # --------------------------------------------------
    cluster_ids = list(set(c["cluster"] for c in chunks if c["cluster"] != -1))
    sample_cluster = random.choice(cluster_ids)

    inspect_cluster(chunks, sample_cluster)

    coh = cluster_coherence(chunks, embeddings, sample_cluster)
    print(f"\nCluster coherence: {coh:.3f}\n")

    # --------------------------------------------------
    # 3. CLUSTER SEPARATION
    # --------------------------------------------------
    cs = centroid_similarity(chunks, embeddings)
    print("Centroid similarity matrix:\n", cs, "\n")

    # --------------------------------------------------
    # 4. QUERY ROUTING
    # --------------------------------------------------
    inspect_query_routing(
        query="What are my responsibilities after purchase?",
        client_id=client_id,
        bot_id=bot_id
    )
