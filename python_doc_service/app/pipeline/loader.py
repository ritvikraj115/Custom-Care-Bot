import os
import json
import faiss
from app.pipeline.hierarchical_index import (
    CLUSTER_INDEX, CLUSTER_META,
    CHUNK_INDEX, CHUNK_META
)
from app.pipeline.storage import INDEX_ROOT

def load_all_indexes():
    if not os.path.exists(INDEX_ROOT):
        return {
            "bots_loaded": 0,
            "chunk_partitions_loaded": 0
        }

    CLUSTER_INDEX.clear()
    CLUSTER_META.clear()
    CHUNK_INDEX.clear()
    CHUNK_META.clear()

    bots_loaded = 0
    partitions_loaded = 0

    for client in os.listdir(INDEX_ROOT):
        if not client.startswith("client_"):
            continue
        cpath = os.path.join(INDEX_ROOT, client)
        if not os.path.isdir(cpath):
            continue

        for bot in os.listdir(cpath):
            if not bot.startswith("bot_"):
                continue
            bpath = os.path.join(cpath, bot)
            if not os.path.isdir(bpath):
                continue

            client_id = client.replace("client_", "", 1)
            bot_id = bot.replace("bot_", "", 1)
            cluster_key = (client_id, bot_id)

            cluster_index_file = os.path.join(bpath, "clusters.faiss")
            cluster_meta_file = os.path.join(bpath, "clusters.meta.json")

            if not os.path.exists(cluster_index_file) or not os.path.exists(cluster_meta_file):
                continue

            try:
                cindex = faiss.read_index(cluster_index_file)
                with open(cluster_meta_file, "r", encoding="utf-8") as f:
                    cmeta = json.load(f)
            except Exception:
                continue

            CLUSTER_INDEX[cluster_key] = cindex
            CLUSTER_META[cluster_key] = cmeta if isinstance(cmeta, list) else []
            bots_loaded += 1

            chunks_dir = os.path.join(bpath, "chunks")
            if not os.path.isdir(chunks_dir):
                continue

            for entry in os.listdir(chunks_dir):
                if not entry.endswith(".meta.json"):
                    continue
                meta_file = os.path.join(chunks_dir, entry)
                index_file = meta_file.replace(".meta.json", ".faiss")
                if not os.path.exists(index_file):
                    continue

                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    cluster_id = payload.get("cluster_id")
                    chunks_meta = payload.get("chunks", [])
                    idx = faiss.read_index(index_file)
                except Exception:
                    continue

                if cluster_id is None:
                    continue

                chunk_key = (client_id, bot_id, cluster_id)
                CHUNK_INDEX[chunk_key] = idx
                CHUNK_META[chunk_key] = chunks_meta if isinstance(chunks_meta, list) else []
                partitions_loaded += 1

    return {
        "bots_loaded": bots_loaded,
        "chunk_partitions_loaded": partitions_loaded
    }
