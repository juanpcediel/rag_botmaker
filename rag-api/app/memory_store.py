from app.chat_memory import ChatMemory

_sessions = {}

def get_session_memory(session_id: str) -> ChatMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ChatMemory()
    return _sessions[session_id]
