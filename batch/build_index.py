import os
import json
import pickle
import pandas as pd
import faiss
from pathlib import Path
from dotenv import load_dotenv

from batch.chunking import build_chunks
from batch.embeddings_local import embed_texts

load_dotenv()

FAISS_LOCAL_DIR = Path(os.getenv("FAISS_LOCAL_DIR", "./artifacts/faiss"))
FAISS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = FAISS_LOCAL_DIR / "Matriz_final_1.csv"


def main():
    print("Leyendo CSV...")
    df = pd.read_csv(CSV_PATH, sep="|", dtype=str).fillna("")
    print("Chunking...")
    chunks = []
    for _, row in df.iterrows():
        chunks.extend(build_chunks(row))

    texts = [c["text"] for c in chunks]

    print("Generando embeddings...")
    vectors = embed_texts(texts)
    dim = vectors.shape[1]

    print("Construyendo índice FAISS...")
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(vectors)

    faiss.write_index(index, str(FAISS_LOCAL_DIR / "products.index"))

    with open(FAISS_LOCAL_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    with open(FAISS_LOCAL_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "chunks": len(chunks),
            "embedding_dim": dim,
            "model": os.getenv("EMBED_MODEL_NAME")
        }, f, indent=2)

    print("✅ Índice FAISS creado correctamente")


if __name__ == "__main__":
    main()
