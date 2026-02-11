import json
import numpy as np

from app.redis.client import redis_client
from app.utils.hashing import stable_hash
from app.cache import keys
from app.memory.ttl import EMBED_CACHE_TTL

def get_cached_embedding(text: str):
    key = keys.embedding(stable_hash(text))
    data = redis_client.get(key)

    if not data:
        return None
    
    return np.array(json.loads(data), dtype='float32')


def set_cache_embedding(text:str, vector:np.ndarray):
    key = keys.embedding(stable_hash(text))

    redis_client.setex(
        key,
        EMBED_CACHE_TTL,
        json.dumps(vector.tolist())
    )