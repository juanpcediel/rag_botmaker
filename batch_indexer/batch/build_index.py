import os
import json
import pickle
import boto3
import pandas as pd
import faiss
from pathlib import Path
from dotenv import load_dotenv

from batch_indexer.chunking import build_chunks
from batch_indexer.embeddings_local import embed_texts

load_dotenv()

FAISS_LOCAL_DIR = Path(os.getenv("FAISS_LOCAL_DIR", "./artifacts/faiss"))
FAISS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILENAME = os.getenv('DATA_FILENAME', 'datos_endpoint.parquet')
DATA_PATH = FAISS_LOCAL_DIR / DATA_FILENAME

S3_BUCKET = os.getenv('S3_BUCKET')
S3_PREFIX = os.getenv('S3_PREFIX', '')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# S3 download logic
def download_from_s3():
    if not S3_BUCKET:
        raise RuntimeError('S3_BUCKET environment variable is no set yet...')
    
    s3_key = f"{S3_PREFIX.rstrip('/')}/{DATA_FILENAME}" if S3_PREFIX else DATA_FILENAME
    print(f"Downloading S3: s3://{S3_BUCKET}/{s3_key}")

    s3 = boto3.client('s3', region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, s3_key, str(DATA_PATH))
    print("=== Download complete ===")


def main():
    print('=== Batch indexer started ===')

    # Check dataset presence
    if not DATA_PATH.exists():
        print(f'Dataset not found locally {DATA_PATH}')
        download_from_s3()
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset cannot be get even after S3 download attempt: {DATA_PATH}')
    
    print(f'Reading dataset from {DATA_PATH}')
    df =  pd.read_parquet(DATA_PATH)

    # Chunking
    print("Chunking ...")
    chunks = []
    for _, row in df.iterrows():
        chunks.extend(build_chunks(row))

    if not chunks:
        raise RuntimeError('No chunks were created from the dataset.')
    
    texts = [c["text"] for c in chunks]

    # Embeddings
    print("=== Generating embeddings ===")
    vectors = embed_texts(texts)
    dim = vectors.shape[1]

    # FAISS index building
    print("=== Building FAISS index ===")
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(vectors)

    # Save index and metadata, atomic operation
    tmp_index = FAISS_LOCAL_DIR / 'products.index.tmp'
    final_index = FAISS_LOCAL_DIR / 'products.index'

    tmp_meta = FAISS_LOCAL_DIR / 'metadata.pkl.tmp'
    final_meta = FAISS_LOCAL_DIR / 'metadata.pkl'

    faiss.write_index(index, str(tmp_index))
    tmp_index.replace(final_index)

    with open(tmp_meta, "wb") as f:
        pickle.dump(chunks, f)
    tmp_meta.replace(final_meta)

    # Manifest file
    with open(FAISS_LOCAL_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "chunks": len(chunks),
                "embedding_dim": dim,
                "dataset": DATA_FILENAME,
                "s3_bucket": S3_BUCKET,
                "s3_prefix": S3_PREFIX,
                "model": os.getenv("EMBED_MODEL_NAME")
            }, 
            f, 
            indent=2
        )

    print("=== FAISS index and metadata saved successfully ===")


if __name__ == "__main__":
    main()
