FROM python:3.10-slim

# Evita bytecode y logs innecesarios
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias mínimas para wheels (NO build-essential)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos requirements primero para cache eficiente
COPY requirements.txt .

# Actualizamos pip y instalamos dependencias CPU-only
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos solo el código necesario
COPY . .

EXPOSE 8001

CMD ["uvicorn", "batch.api:app", "--host", "0.0.0.0", "--port", "8001"]
