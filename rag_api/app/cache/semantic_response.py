import json
import numpy as np
import logging

from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.redis.client import redis_client
from app.embeddings.client import embed_texts

# Basic log configuration to view console output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RedisSemanticSearch")

INDEX_NAME = 'qa_semantic_idx'
# It is the size of the vector that generates the model MiniLM
VECTOR_DIM = 384
VECTOR_FIELD = 'embeddings'


# Create index - run once

def create_index():
    try:
        redis_client.ft(INDEX_NAME).info()
        return
    except:
        pass

    schema = (
        TextField('answer'),
        VectorField(
            VECTOR_FIELD,
            'HNSW',
            {
                'TYPE':'FLOAT32',
                'DIM': VECTOR_DIM,
                'DISTANCE_METRIC': 'COSINE',
                'M':16,
                'EF_CONSTRUCTION': 200
            }
        )
    )

    redis_client.ft(INDEX_NAME).create_index(
        schema,
        definition = IndexDefinition(prefix=['qa:'],
                                    index_type=IndexType.HASH)
    )


# Save the QA pairs
def save_semantic_response(question:str, answer:str):
    vec = embed_texts([question])[0]
    key = f'qa:{stable_hash(question)}'

    redis_client.hset(key,
                      mapping={
                          'answer':answer,
                          VECTOR_FIELD: vec.tobytes()
                      }
                      )

# Search similar
# K=3 because we want to see the distance between vectors
def search_semantic_response(question:str, k=3, threshold = 0.9):
    vec = embed_texts([question])[0]
    
    # We calculate the maximum allowed distance
    # Similarity = 0.9, Distance = 0.1
    max_distance = 1 - threshold

    # We prepare the query (we will use the normal version to see EVERYTHING that arrives)
    q = (
        Query(f"@{VECTOR_FIELD}:[VECTOR_RANGE {max_distance} $vec]=>[KNN {k} @{VECTOR_FIELD} $vec AS score]")
        .sort_by('score')
        .return_fields('answer', 'score')
        .dialect(2)
    )

    res = redis_client.ft(INDEX_NAME).search(q, {'vec': vec.tobytes()})


    if not res.docs:
        logger.info(f"🔍 Pregunta: '{question}' | ❌ No se encontraron resultados.")
        return None
    
    # Block of logs


    # Since Redis has already filtered, the first result is guaranted to be valid
    return res.docs[0].answer

# SEARCH SEMANTIC RESPONSE 
# 
# import logging

# # Configuración básica de logs para ver la salida en consola
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("RedisSemanticSearch")

# def search_semantic_response(question: str, k=3, threshold=0.90): # Subimos k a 3 para comparar
#     vec = embed_texts([question])[0]
    
#     # Preparamos la query (usaremos la versión normal para ver todo lo que llega)
#     q = (
#         Query(f"*=>[KNN {k} @{VECTOR_FIELD} $vec AS score]")
#         .sort_by("score")
#         .return_fields("answer", "score")
#         .dialect(2)
#     )

#     res = redis_client.ft(INDEX_NAME).search(q, {"vec": vec.tobytes()})

#     if not res.docs:
#         logger.info(f"🔍 Pregunta: '{question}' | ❌ No se encontraron resultados.")
#         return None

#     # --- BLOQUE DE LOGS ---
#     logger.info(f"🔍 Analizando pregunta: '{question}'")
#     for i, doc in enumerate(res.docs):
#         distancia = float(doc.score)
#         similitud = 1 - distancia
#         logger.info(
#             f"  Resultado #{i+1}: "
#             f"Distancia: {distancia:.4f} | "
#             f"Similitud: {similitud:.4f} | "
#             f"Respuesta: {doc.answer[:50]}..."
#         )
#     # -----------------------

#     # Filtro de lógica
#     doc = res.docs[0]
#     similarity = 1 - float(doc.score)

#     if similarity < threshold:
#         logger.warning(f"⚠️ El mejor resultado ({similarity:.4f}) no superó el umbral de {threshold}")
#         return None

#     return doc.answer