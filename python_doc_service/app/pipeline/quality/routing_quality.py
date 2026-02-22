from app.pipeline.hierarchical_index import query_hierarchical
from app.pipeline.embed import model

def inspect_query_routing(
    query,
    client_id,
    bot_id,
    top_clusters=2,
    top_chunks=5
):
    print("\n================ QUERY ROUTING ================\n")
    print("QUERY:", query, "\n")

    q_emb = model.encode(query, normalize_embeddings=True)

    results = query_hierarchical(
        q_emb,
        client_id=client_id,
        bot_id=bot_id,
        top_clusters=top_clusters,
        top_chunks=top_chunks,
        query_text=query,
    )

    for r in results:
        print(f"[Cluster {r['cluster']} | Topic: {r['topic']}]")
        print(r["text"][:300])
        print("-" * 60)
