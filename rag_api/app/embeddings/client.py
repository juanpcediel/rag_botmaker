import requests
import numpy as np

BATCH_INDEXER_URL = "http://batch_indexer:8001/embed"

def embed_texts(texts: list[str]) ->np.ndarray:
    try:
        response = requests.post(
            BATCH_INDEXER_URL,
            json={"texts": texts},
            timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError("Failed to get embeddings from batch_indexer") from e

    data = response.json()

    if "embeddings" not in data:
        raise RuntimeError(
            f"Invalid response from batch_indexer: {data}"
        )
    return np.array(data["embeddings"], dtype="float32")
