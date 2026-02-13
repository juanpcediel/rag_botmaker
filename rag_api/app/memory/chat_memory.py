import faiss
import time

from app.embeddings.client import embed_texts
from app.memory.store import get_history, save_history

class ChatMemory:
    # Constructor
    def __init__(self, session_id: str):
        self.session_id = session_id
        # Load Redis history
        self.turns = get_history(session_id)
        self.index = None

        # rebuild index if history exists
        if self.turns:
            self._build_index()
    
    # Build FAISS index
    def _build_index(self):
        texts = [text for _, text in self.turns]
        vecs = embed_texts(texts)
        dim = vecs.shape[1]

        self.index = faiss.IndexHNSWFlat(
            dim,
            16,
            faiss.METRIC_INNER_PRODUCT
        )
        # FAISS class SearchParametersHNSW 
        self.index.hnsw.efSearch = 64
        self.index.add(vecs)

    # Add turns
    def add_turn(self, role:str, text:str):
        self.turns.append((role,text))

        #Persist in Redis
        save_history(self.session_id, self.turns)

        #Update local index
        vec = embed_texts([text])

        if self.index is None:
            self._build_index()
        else:
            self.index.add(vec)

    # Last Messages
    def last_n(self, n:int = 6):
        return self.turns[-n:]
    
    # Semantic retrieval
    def retrieve_relevant(self, query:str, k: int = 4):
        if self.index is None:
            return []
        
        qv = embed_texts([query])

        _, idxs = self.index.search(qv, min(k,len(self.turns)))

        return [self.turns[i] for i in idxs[0] if i>=0]
        

        


# class ChatMemory:
#     def __init__(self):
#         self.turns = []            # [(role, text)]
#         self.index = None          # FAISS index
#         self.created_at = time.time()
#         self.last_used = time.time()

#     def _ensure_index(self, dim: int):
#         if self.index is None:
#             self.index = faiss.IndexHNSWFlat(
#                 dim,
#                 16,
#                 faiss.METRIC_INNER_PRODUCT
#             )
#             self.index.hnsw.efSearch = 64

#     def add_turn(self, role: str, text: str):
#         self.last_used = time.time()
#         self.turns.append((role, text))

#         vec = embed_texts([text])
#         dim = vec.shape[1]
#         self._ensure_index(dim)

#         self.index.add(vec)

#     def last_n(self, n: int = 6):
#         return self.turns[-n:]

#     def retrieve_relevant(self, query: str, k: int = 4):
#         if self.index is None:
#             return []

#         qv = embed_texts([query])
#         _, idxs = self.index.search(qv, min(k, len(self.turns)))

#         results = []
#         for i in idxs[0]:
#             if i >= 0:
#                 results.append(self.turns[i])

#         return results
