import logging
import time

from app.retrieval.products import retrieve_products
# from app.cache.response import get_cached_response, set_cached_response
from app.cache.semantic_response import (search_semantic_response, save_semantic_response)
from app.rag.prompt import PROMPT_TEMPLATE

# logger initiation
logger = logging.getLogger("rag.pipeline")

# Format helper
def format_chat_history(turns):
    return "\n".join(
        f"{role}: {text}" for role, text in turns
    )


# Main pipeline
def generate_answer(store, question, memory, llm_call):

    # extract the value from the 'session' object
    session = getattr(memory, 'session_id', 'unknown')

    # Response cache first
    # cached = get_cached_response(question)
    cached = search_semantic_response(question)
    
    if cached:
        logger.info(f"[SESSION={session}] ✅ SEMANTIC_CACHE_HIT")
        t0 = time.perf_counter()
        # Retrieval
        context, products = retrieve_products(store, question)
        
        dt = (time.perf_counter() - t0) * 1000

        logger.debug(f"[SESSION={session}] retrieval_time={dt:.1f}ms")

        memory.add_turn("user", question)
        memory.add_turn("assistant", cached)
        

        return cached, products
    
    logger.info(f"[SESSION={session}] ❌ SEMANTIC_CACHE_HIT -> calling LLM")

    # Retrieval, only if cache doesn't exists
    t0 = time.perf_counter()
    context, products = retrieve_products(store, question)
    retrieval_dt = (time.perf_counter() - t0) * 1000

    logger.debug(f"[SESSION={session}] retrieval_time={retrieval_dt:.1f}ms")

    # Memory context
    recent = memory.last_n(6)
    relevant = memory.retrieve_relevant(question, k=3)

    seen = set()
    merged = []

    for t in relevant + recent:
        if t not in seen:
            seen.add(t)
            merged.append(t)

    chat_history = format_chat_history(merged)

    # Prompt
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        chat_history=chat_history,
        question=question
    )
    
    # # Search in cache
    # cached = get_cached_response(question)
    # if cached:
    #     return cached, products
    
    # LLM call
    t0 = time.perf_counter()
    answer = llm_call(prompt)
    llm_dt = (time.perf_counter() - t0) * 1000

    logger.info(f"[SESSION={session}] 🤖 LLM_CALL time={llm_dt:.1f}ms")

    # Modify cache: save cache + memory
    # set_cached_response(question, answer)
    save_semantic_response(question, answer)


    memory.add_turn("user", question)
    memory.add_turn("assistant", answer)

    return answer, products
