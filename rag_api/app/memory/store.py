import json

from app.redis.client import redis_client
from app.cache import keys
from app.memory.ttl import SESSION_HISTORY_TTL

def _key(session_id):
    return f"session: {session_id}"

def get_history(session_id: str) -> list[tuple[str,str]]:
    """
    Docstring for get_history
    
    :param session_id: Description
    :type session_id: str
    :return: Description
    :rtype: list[tuple[str, str]]

    Returns: list of (role, text)
    """
    raw = redis_client.get(keys.session_history(session_id))
    if not raw:
        return []
    data = json.loads(raw)

    # Normalize to list[tuple[str, str]]
    return [(item[0], item[1]) for item in data]


def save_history (session_id:str, turns:list[tuple[str, str]]) -> None:
    """
    Docstring for save_history
    
    :param session_id: Description
    :type session_id: str
    :param turn: Description
    :type turn: list[tuple[str, str]]
    """

    redis_client.setex(
        keys.session_history(session_id),
        SESSION_HISTORY_TTL,
        json.dumps(turns)
    )