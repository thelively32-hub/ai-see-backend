# ===== Base =====
FROM python:3.11-slim

# Config básica
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ===== SO deps útiles (ffmpeg para pydub, libgl para OpenCV, etc.) =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libimage-exiftool-perl \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gcc \
    g++ \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    ca-certificates \
    curl \
  && rm -rf /var/lib/apt/lists/*

# ===== Python deps (cache-friendly) =====
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ===== Playwright con dependencias de SO =====
# (así no tienes que listar a mano todas las libs del navegador)
RUN python -m playwright install --with-deps chromium

# ===== Código =====
COPY . .

# Render asigna $PORT; usa 10000 por defecto local
EXPOSE 10000

# ===== Start =====
# Usa $PORT de Render si existe; si no, 10000 para local
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]

