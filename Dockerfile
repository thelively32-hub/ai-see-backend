# ===== Base =====
FROM python:3.11-slim

# Config básica
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=0

WORKDIR /app

# ===== SO deps necesarias =====
# ffmpeg: pydub | libgl/libglib: opencv | ca-certificates/curl: playwright con --with-deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
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
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# ===== Playwright con dependencias de sistema =====
RUN python -m playwright install --with-deps chromium

# ===== Código de la app =====
COPY . .

# ===== Puerto =====
EXPOSE 10000

# ===== Start =====
# Usa $PORT de Render si existe; si no, 10000 para local
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]



