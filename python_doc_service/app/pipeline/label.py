from sklearn.feature_extraction.text import TfidfVectorizer
from app.pipeline.logger import get_logger
log = get_logger("label")


def label_clusters(chunks):
    groups = {}

    for c in chunks:
        if c["cluster"] == -1:
            continue
        groups.setdefault(c["cluster"], []).append(c["text"])

    labels = {}

    for cid, texts in groups.items():
        tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=300
        )
        X = tfidf.fit_transform(texts)
        scores = X.sum(axis=0).A1
        terms = tfidf.get_feature_names_out()

        top = [terms[i] for i in scores.argsort()[-5:][::-1]]
        labels[cid] = ", ".join(top)
        log.info(f"Labeling {len(labels)} clusters")


    return labels
