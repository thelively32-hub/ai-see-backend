#!/usr/bin/env bash
set -Eeuo pipefail

# Opcional: indica a pydub dónde está ffmpeg si tienes imageio-ffmpeg instalado.
FF_BIN="$(python - <<'PY'
try:
    import imageio_ffmpeg
    print(imageio_ffmpeg.get_ffmpeg_exe())
except Exception:
    print("")
PY
)"
if [[ -n "$FF_BIN" ]]; then
  export FFMPEG_BINARY="$FF_BIN"
fi

# Usa el puerto que provee Render; por defecto 10000 en local.
exec python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-10000}"
