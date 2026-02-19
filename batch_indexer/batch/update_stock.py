import boto3
import pandas as pd
import os
from pathlib import Path

from batch.redis_client import redis_client
from batch.data_processing import parquet_data_processing


S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "")

DATA_FILENAME = os.getenv('DATA_FILENAME', 'datos_endpoint.parquet')
FAISS_LOCAL_DIR = Path(os.getenv('FAISS_LOCAL_DIR', './artifacts/faiss'))
DATA_PATH = FAISS_LOCAL_DIR / DATA_FILENAME

def update_stock():
    if not S3_BUCKET:
        print('S3_BUCKET not configured')
        return

    s3 = boto3.client("s3")

    key = f"{S3_PREFIX.rstrip('/')}/{DATA_FILENAME}" if S3_PREFIX else DATA_FILENAME

    # Download and overwrite local file
    s3.download_file(S3_BUCKET, key, str(DATA_PATH))
    
    df_raw = pd.read_parquet(DATA_PATH)

    # Clean data (*** To be reviewed ***)
    df_stock = parquet_data_processing(df_raw)

    # Remove old stock
    delete_pipe = redis_client.pipeline()
    for k in redis_client.scan_iter("stock:*"):
        delete_pipe.delete(k)
    delete_pipe.execute()

    # Save only available stock
    pipe = redis_client.pipeline()
    for _, row in df_stock.iterrows():
        pipe.set(f"stock:{row['sku']}", int(row["Inventario"]))
    pipe.execute()

    print(f"Stock updated. Active SKUs: {len(df_stock)}")
