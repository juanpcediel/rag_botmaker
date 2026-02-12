from collections import defaultdict
from app.embeddings.client import embed_texts


def retrieve_products(store, query, top_k_products=5, overfetch=30):

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
            "sku": best["sku"],
            "title": best["title"],
            "image": best["image"],
            "link": best["link"],
            "price": best.get("price"),
            "stock": best.get("stock"),
        })

        context_blocks.append(merged_text)

    context = "\n\n---\n\n".join(context_blocks)

    return context, products

# old retrieve_products:

# def retrieve_product_context(store, query, top_k_products=5, overfetch=30):
#     qv = embed_texts([query])
#     _, idxs = store.index.search(qv, overfetch)
#     grouped = defaultdict(list)
#     order = []

#     for i in idxs[0]:
#         if i < 0:
#             continue
#         item = store.metadata[i]
#         pid = item["product_id"]
#         grouped[pid].append(item)
#         if pid not in order:
#             order.append(pid)
#     context_blocks = []
#     products = []

#     for pid in order[:top_k_products]:
#         items = grouped[pid]
#         merged_text = "\n\n".join(x["text"] for x in items)

#         best = items[0]
#         # Important add sku from metadata index vector

#         products.append({
#             "title": best["title"],
#             "image": best["image"],
#             "link": best["link"]
#         })

#         context_blocks.append(merged_text)

#     context = "\n\n---\n\n".join(context_blocks)
#     return context, products