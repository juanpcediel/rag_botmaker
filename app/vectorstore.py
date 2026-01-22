import faiss
import pickle
from app.config import settings

class VectorStore:
    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata

def load_vectorstore() -> VectorStore:
    index_path = settings.FAISS_LOCAL_DIR / "products.index"
    meta_path = settings.FAISS_LOCAL_DIR / "metadata.pkl"

    if not index_path.exists():
        raise RuntimeError(
            "No existe índice FAISS. Ejecuta primero: python -m batch.build_index"
        )

    index = faiss.read_index(str(index_path))
    index.hnsw.efSearch = 128

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return VectorStore(index, metadata)
