from app.pipeline.logger import get_logger
from app.pipeline.semantic_embedder import get_embedder
log = get_logger("embed")


model = get_embedder()

def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    log.info(f"Embedding {len(texts)} chunks")

    return model.encode(texts, normalize_embeddings=True)
    
