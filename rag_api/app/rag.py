from collections import defaultdict
from app.embeddings_client import embed_texts
from app.config import settings
from app.prompt import PROMPT_TEMPLATE


def retrieve_product_context(store, query, top_k_products=5, overfetch=30):
    qv = embed_texts([query])
    _, idxs = store.index.search(qv, overfetch)
    grouped = defaultdict(list)
    order = []

    for i in idxs[0]:
        if i < 0:
            continue
        item = store.metadata[i]
        pid = item["product_id"]
        grouped[pid].append(item)
        if pid not in order:
            order.append(pid)
    context_blocks = []
    products = []

    for pid in order[:top_k_products]:
        items = grouped[pid]
        merged_text = "\n\n".join(x["text"] for x in items)

        best = items[0]
        products.append({
            "title": best["title"],
            "image": best["image"],
            "link": best["link"]
        })

        context_blocks.append(merged_text)

    context = "\n\n---\n\n".join(context_blocks)
    return context, products
    


def format_chat_history(turns):
    return "\n".join(
        f"{role}: {text}" for role, text in turns
    )


def generate_answer(store, question, memory, llm_call):
    context, products = retrieve_product_context(store, question)

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
