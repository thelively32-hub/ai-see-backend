# Usa una imagen oficial de Python como base
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala las dependencias del sistema operativo que necesitamos
# - ffmpeg para pydub
# - libimage-exiftool-perl es el paquete para exiftool
# - libgl1 y libglib2.0-0 para opencv-headless
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libimage-exiftool-perl \
    libgl1 \
    libglib2.0-0 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copia el archivo de requerimientos
COPY requirements.txt .

# Instala las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Instala los navegadores de Playwright
RUN playwright install

# Copia el resto del código de tu aplicación
COPY . .

# Expone el puerto que usará la aplicación. Render usará este puerto internamente.
EXPOSE 10000

# Comando para ejecutar la aplicación. Usamos un puerto fijo.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
