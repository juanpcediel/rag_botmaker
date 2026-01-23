import faiss
import time
from batch_indexer.embeddings_local import embed_texts


class ChatMemory:
    def __init__(self):
        self.turns = []            # [(role, text)]
        self.index = None          # FAISS index
        self.created_at = time.time()
        self.last_used = time.time()

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexHNSWFlat(
                dim,
                16,
                faiss.METRIC_INNER_PRODUCT
            )
            self.index.hnsw.efSearch = 64

    def add_turn(self, role: str, text: str):
        self.last_used = time.time()
        self.turns.append((role, text))

        vec = embed_texts([text])
        dim = vec.shape[1]
        self._ensure_index(dim)

        self.index.add(vec)

    def last_n(self, n: int = 6):
        return self.turns[-n:]

    def retrieve_relevant(self, query: str, k: int = 4):
        if self.index is None:
            return []

        qv = embed_texts([query])
        _, idxs = self.index.search(qv, min(k, len(self.turns)))

        results = []
        for i in idxs[0]:
            if i >= 0:
                results.append(self.turns[i])

        return results
