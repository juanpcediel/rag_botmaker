def clean(x: str) -> str:
    return (x or "").strip()


def build_chunks(row: dict) -> list[dict]:
    product_id = row.get("Link") or row.get("Nombre_producto")

    title = clean(row.get("Nombre_producto"))
    category = clean(row.get("Categoria"))
    brand = clean(row.get("Marca"))
    keywords = clean(row.get("Keywords"))
    description = clean(row.get("Descripcion_producto"))
    meta = clean(row.get("MetaTagDescription"))
    price = clean(row.get("Precio"))
    size = clean(row.get("Talla"))

    image = clean(row.get("Imagen"))
    link = clean(row.get("Link"))

    chunks = []

    # Chunk resumen (búsquedas generales)
    chunks.append({
        "product_id": product_id,
        "title": title,
        "text": f"""Producto: {title}
                Categoría: {category}
                Marca: {brand}
                Keywords: {keywords}
                Precio: {price}
                Talla: {size}""".strip(),
        "image": image,
        "link": link
    })

    # Chunk descripción (búsquedas semánticas)
    long_desc = "\n".join([x for x in [description, meta] if x])
    if long_desc:
        chunks.append({
            "product_id": product_id,
            "title": title,
            "text": f"""Producto: {title}
                Descripción:
                {long_desc}""".strip(),
            "image": image,
            "link": link
        })

    return chunks
