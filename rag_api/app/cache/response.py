import json

from app.redis.client import redis_client
from app.utils.hashing import stable_hash
from app.cache import keys
from app.memory.ttl import RESPONSE_CACHE_TTL

def get_cached_response(question: str):
    key = keys.response(stable_hash(question))
    return redis_client.get(key)

def set_cached_response(question:str, answer:str):
    key = keys.response(stable_hash(question))

    redis_client.setex(
        key,
        RESPONSE_CACHE_TTL,
        answer
    )

