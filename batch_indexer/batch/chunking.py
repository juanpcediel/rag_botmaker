def clean(x):
    return (str(x) if x is not None else "").strip()


def build_chunks(row: dict) -> list[dict]:

    sku = clean(row.get("sku"))
    product_id = sku            

    title = clean(row.get("Nombre_producto"))
    description = clean(row.get("Descripcion_producto"))
    keywords = clean(row.get("Keywords"))
    meta = clean(row.get("MetaTagDescription"))
    category = clean(row.get("Nombre_Categoria"))
    brand = clean(row.get("Marca"))
    size = clean(row.get("Talla"))
    color = clean(row.get("Color"))

    image = clean(row.get("Imagen_url"))
    link = clean(row.get("Link"))

    price = row.get("Precio")
    stock = row.get("Inventario")

    chunks = []

    # Chunk semántico natural
    
    semantic_text = " ".join(filter(None, [
        title,
        f"marca {brand}" if brand else "",
        f"categoria {category}" if category else "",
        f"color {color}" if color else "",
        f"talla {size}" if size else "",
        keywords,
        description,
        meta
    ]))

    chunks.append({
        "product_id": product_id,
        "sku": sku,               
        "title": title,
        "text": semantic_text,
        "image": image,
        "link": link,
        "price": price,
        "stock": stock
    })

    
    # Chunk corto keywords
    
    short_text = " ".join(filter(None, [
        title, brand, category, color, size, keywords
    ]))

    chunks.append({
        "product_id": product_id,
        "sku": sku,
        "title": title,
        "text": short_text,
        "image": image,
        "link": link,
        "price": price,
        "stock": stock
    })

    return chunks