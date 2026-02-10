from collections import defaultdict

from rag_api.app.embeddings.client import embed_texts
from app.retrieval import retrieve_products

from app.config import settings
from rag_api.app.rag.prompt import PROMPT_TEMPLATE

def format_chat_history(turns):
    return "\n".join(
        f"{role}: {text}" for role, text in turns
    )


def generate_answer(store, question, memory, llm_call):
    context, products = retrieve_products(store, question)

    recent = memory.last_n(6)
    relevant = memory.retrieve_relevant(question, k=3)

    seen = set()
    merged = []
    for t in relevant + recent:
        if t not in seen:
            seen.add(t)
            merged.append(t)

    chat_history = format_chat_history(merged)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        chat_history=chat_history,
        question=question
    )

    answer = llm_call(prompt)

    memory.add_turn("user", question)
    memory.add_turn("assistant", answer)

    return answer, products
