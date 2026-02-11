def session_history (session_id: str) -> str:
    return f"session:{session_id}:history"

def embedding (text_hash:str)-> str:
    return f"embed:{text_hash}"

def response(text_hash: str) -> str:
    return f"qa: {text_hash}"

