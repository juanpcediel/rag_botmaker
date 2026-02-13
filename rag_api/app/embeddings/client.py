import requests
import numpy as np

from app.embeddings.cache import get_cached_embedding, set_cache_embedding

BATCH_INDEXER_URL = "http://batch_indexer:8001/embed"

# Actual call to the batch_indexer service
def _call_batch_indexer(texts:list[str]) ->np.ndarray:
    try:
        response = requests.post(
            BATCH_INDEXER_URL,
            json={"texts": texts},
            timeout=10
        )
        response.raise_for_status()
        
    except requests.RequestException as e:
        raise RuntimeError("*** Failed to get embeddings from batch_indexer ***") from e

    data = response.json()

    if 'embeddings' not in data:
        raise RuntimeError(f'*** Invalid response from batch_indexer service, check the following data: {data} ***')
    
    return np.array(data['embeddings'], dtype='float32')

# Public function using cache
def embed_texts(texts: list[str]) ->np.ndarray:
    results = []
    missing_texts = []
    missing_positions = []

    # Search in cache
    for i, text in enumerate(texts):
        cached = get_cached_embedding(text)

        if cached is not None:
            results.append(cached)
        else:
            results.append(None)
            missing_texts.append(text)
            missing_positions.append(i)
    
    # Order only the missing items
    if missing_texts:
        vectors = _call_batch_indexer(missing_texts)

        # Save in cache
        for pos, text, vec in zip(missing_positions, missing_texts, vectors):
            set_cache_embedding(text, vec)
            results[pos] = vec
    
    # Return all items together
    return np.vstack(results).astype('float32')

