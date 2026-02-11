import hashlib

def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
