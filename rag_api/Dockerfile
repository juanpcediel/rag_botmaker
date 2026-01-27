FROM python:3.11-slim

# Evita .pyc y fuerza logs en tiempo real (clave en contenedores)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias del sistema (FAISS / numpy / torch-friendly)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalamos dependencias primero para aprovechar cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos solo el código del servicio
COPY batch ./batch

EXPOSE 8001

# Exec form → señales correctas (SIGTERM, SIGINT)
CMD ["uvicorn", "batch.api:app", "--host", "0.0.0.0", "--port", "8001"]
