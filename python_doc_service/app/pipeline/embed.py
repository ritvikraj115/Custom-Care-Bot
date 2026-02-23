from sentence_transformers import SentenceTransformer
from app.pipeline.logger import get_logger
log = get_logger("embed")


model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    log.info(f"Embedding {len(texts)} chunks")

    return model.encode(texts, normalize_embeddings=True)
    
