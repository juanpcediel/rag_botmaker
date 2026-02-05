import os
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    model = get_model()

    batch_size = 64
    total = len(texts)
    vectors = []

    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]

        vec = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        vectors.append(vec)

        print(f"Embedding progress: {min(i+batch_size, total)}/{total}")

    return np.vstack(vectors).astype("float32")



# def embed_texts(texts: list[str]) -> np.ndarray:
#     model = get_model()
#     embeddings = model.encode(
#         texts,
#         convert_to_numpy=True,
#         normalize_embeddings=True,
#         batch_size=64,
#         show_progress_bar=True
#     )
#     return embeddings.astype("float32")