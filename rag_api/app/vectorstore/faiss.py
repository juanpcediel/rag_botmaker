import faiss
import pickle
import time
from pathlib import Path

from app.config import settings


class VectorStore:
    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata


# -------------------------
# WAIT FOR INDEX (robusto)
# -------------------------
def wait_for_index(timeout=600):
    index_dir = Path(settings.FAISS_LOCAL_DIR)

    start = time.time()

    while True:
        index_file = index_dir / "products.index"
        meta_file = index_dir / "metadata.pkl"

        if index_file.exists() and meta_file.exists():
            break

        if time.time() - start > timeout:
            raise RuntimeError("FAISS index not created in time")

        print("⏳ Waiting for FAISS index...")
        time.sleep(2)


# -------------------------
# LOAD STORE
# -------------------------
def load_vectorstore() -> VectorStore:
    wait_for_index()

    index_path = settings.FAISS_LOCAL_DIR / "products.index"
    meta_path = settings.FAISS_LOCAL_DIR / "metadata.pkl"

    index = faiss.read_index(str(index_path))
    index.hnsw.efSearch = 128

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return VectorStore(index, metadata)
