import asyncio
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from batch.build_index import build_index
from batch.update_stock import update_stock

app = FastAPI()

# Faiss index rebuild
REINDEX_INTERVAL = 12 * 60 * 60

# Stock update 
STOCK_REFRESH_INTERVAL = 60

reindex_lock = asyncio.Lock()
model = None  # lazy load


# -------------------------
# MODEL LOAD (startup)
# -------------------------
@app.on_event("startup")
async def load_model():
    global model
    print("*** Loading embedding model... ***")
    loop = asyncio.get_running_loop()
    model = await loop.run_in_executor(
        None,
        lambda: SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
    )
    print("*** Model loaded ***")


# -------------------------
# Core async rebuild index
# -------------------------
async def run_reindex():
    if reindex_lock.locked():
        print("*** Reindex already running, skipping... ***")
        return

    async with reindex_lock:
        print(f"*** Reindex started at {datetime.now()} ***")

        loop = asyncio.get_running_loop()

        # NON BLOCKING (important)
        await loop.run_in_executor(None, build_index)

        print(f"*** Reindex finished at {datetime.now()} ***")

# -------------------------
# Scheduler 12h rebuild faiss index
# -------------------------
async def periodic_reindex():
    while True:
        await asyncio.sleep(REINDEX_INTERVAL)
        await run_reindex()

# -------------------------
# Scheduler 1 min update stock
# -------------------------
async def periodic_stock_refresh():
    while True:
        await asyncio.sleep(STOCK_REFRESH_INTERVAL)
        print('*** Updating Stock... ***')
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, update_stock)
        print('*** Stock updated ***')

# -------------------------
# Startup behavior
# -------------------------
@app.on_event("startup")
async def startup_tasks():
    # Build index once (blocking)
    await run_reindex()

    # Update stock once (blocking)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, update_stock)

    # Start background schedulers
    asyncio.create_task(periodic_reindex())
    asyncio.create_task(periodic_stock_refresh())


# -------------------------
# Manual ENDPOINT
# -------------------------
@app.post("/reindex")
async def manual_reindex():
    asyncio.create_task(run_reindex())
    return {"status": "reindex started in background"}


# -------------------------
# Embeddings ENDPOINT
# -------------------------
class EmbedRequest(BaseModel):
    texts: list[str]


@app.post("/embed")
async def embed(req: EmbedRequest):
    loop = asyncio.get_running_loop()

    vectors = await loop.run_in_executor(
        None,
        lambda: model.encode(req.texts, normalize_embeddings=True),
    )

    return {"embeddings": vectors.tolist()}
