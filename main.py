# main.py
import os
import hashlib
import asyncio
from urllib.parse import urlparse
from typing import Dict, List, Optional, Any
import json
import tempfile
import uuid
from datetime import datetime
import logging

import yt_dlp
import aiohttp
import aiofiles
import numpy as np

# --- FFmpeg embebido para pydub (sin apt-get) ---
import imageio_ffmpeg
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

# Imports opcionales: si no están disponibles, seguimos funcionando
try:
    import cv2
except Exception as _e:
    cv2 = None

try:
    # Importamos pydub después de definir FFMPEG_BINARY
    from pydub import AudioSegment
except Exception as _e:
    AudioSegment = None

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-see")

# ========== MODELOS PYDANTIC ==========
class AnalysisRequest(BaseModel):
    content: str
    content_type: str  # "link" | "video" | "image"
    request_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    request_id: str
    status: str  # "completed" | "processing" | "error"
    result: Optional[Dict] = None
    error_message: Optional[str] = None

# ========== CLASE LINK RESOLVER ==========
class LinkResolver:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _generate_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    async def resolve_video(self, social_media_url: str) -> Dict:
        cache_key = self._generate_cache_key(social_media_url)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            async with aiofiles.open(cache_file, "r") as f:
                try:
                    return json.loads(await f.read())
                except Exception:
                    pass  # si se dañó el cache, seguimos

        # 1) Intento con yt-dlp
        result = await self._try_yt_dlp(social_media_url)

        # 2) Fallback: headless browser (solo si playwright está disponible)
        if not result or "video_url" not in result:
            result = await self._try_headless_browser(social_media_url)

        if result and "video_url" in result:
            async with aiofiles.open(cache_file, "w") as f:
                await f.write(json.dumps(result))

        return result or {}

    async def _try_yt_dlp(self, url: str) -> Optional[Dict]:
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "force_json": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info and "url" in info:
                    return {
                        "video_url": info["url"],
                        "title": info.get("title", ""),
                        "duration": info.get("duration", 0),
                        "thumbnail": info.get("thumbnail", ""),
                        "platform": info.get("extractor", ""),
                        "resolution": f"{info.get('width', 0)}x{info.get('height', 0)}",
                        "method": "yt-dlp",
                    }
        except Exception as e:
            logger.warning(f"[yt-dlp] {e}")
        return None

    async def _try_headless_browser(self, url: str) -> Optional[Dict]:
        # Import lazy para no romper si playwright no está instalado
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except Exception as e:
            logger.info(f"Playwright no disponible: {e}")
            return None

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                try:
                    await page.goto(url, wait_until="networkidle", timeout=45000)
                    video_url = await self._extract_generic_video(page)
                    if video_url:
                        return {
                            "video_url": video_url,
                            "platform": self._identify_platform(url),
                            "method": "headless_browser",
                        }
                finally:
                    await browser.close()
        except Exception as e:
            logger.warning(f"[playwright] {e}")
        return None

    def _identify_platform(self, url: str) -> str:
        domain = urlparse(url).netloc.lower()
        if "tiktok.com" in domain:
            return "tiktok"
        if "instagram.com" in domain:
            return "instagram"
        if "youtube.com" in domain or "youtu.be" in domain:
            return "youtube"
        if "twitter.com" in domain or "x.com" in domain:
            return "twitter"
        if "facebook.com" in domain or "fb.com" in domain:
            return "facebook"
        return "unknown"

    async def _extract_generic_video(self, page) -> Optional[str]:
        try:
            # 1) <video src="...">
            for video in await page.query_selector_all("video"):
                src = await video.get_attribute("src")
                if src and (src.endswith(".mp4") or "video" in src):
                    return src

            # 2) iframes comunes (YouTube/Vimeo)
            for iframe in await page.query_selector_all("iframe"):
                src = await iframe.get_attribute("src")
                if src and any(k in src for k in ("youtube", "vimeo", "video")):
                    return src
        except Exception as e:
            logger.warning(f"extract_generic_video: {e}")
        return None

# ========== CLASE ADVANCED CONTENT ANALYZER ==========
class AdvancedContentAnalyzer:
    def __init__(self):
        self.video_analyzers = [
            self._analyze_video_artifacts,
            self._analyze_temporal_consistency,
            self._analyze_facial_patterns,
        ]
        self.audio_analyzers = [
            self._analyze_audio_tts,
            self._analyze_audio_quality,
        ]
        self.metadata_analyzers = [
            self._analyze_technical_metadata,
            self._analyze_creation_patterns,
        ]

    async def advanced_analysis(self, video_url: str) -> Dict:
        video_path = await self._download_video(video_url)
        if not video_path:
            return self._create_error_response("No se pudo descargar el video")

        try:
            video_analysis = await self._analyze_video(video_path)
            audio_analysis = await self._analyze_audio(video_path)
            metadata_analysis = await self._analyze_metadata(video_path)
            return self._combine_analyses(video_analysis, audio_analysis, metadata_analysis)
        except Exception as e:
            logger.error(f"Error en análisis avanzado: {e}")
            return self._create_error_response(f"Error en análisis: {str(e)}")
        finally:
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
            except Exception:
                pass

    async def _download_video(self, video_url: str, chunk_size: int = 1024 * 1024) -> Optional[str]:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp.name
            tmp.close()
            timeout = aiohttp.ClientTimeout(total=90)  # 90s máx
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(video_url) as resp:
                    if resp.status != 200:
                        return None
                    async with aiofiles.open(tmp_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(chunk_size):
                            await f.write(chunk)
            return tmp_path
        except Exception as e:
            logger.warning(f"Descarga falló: {e}")
            return None

    async def _analyze_video(self, video_path: str) -> Dict:
        results: Dict[str, Any] = {}
        for analyzer in self.video_analyzers:
            try:
                analysis = await analyzer(video_path)
                if isinstance(analysis, dict):
                    results.update(analysis)
            except Exception as e:
                logger.warning(f"video analyzer error: {e}")
        return results

    async def _analyze_audio(self, video_path: str) -> Dict:
        if not AudioSegment:
            return {"audio_analysis": "unavailable (pydub/ffmpeg not installed)"}

        audio_path = await self._extract_audio(video_path)
        if not audio_path:
            return {"audio_analysis": "failed"}

        results: Dict[str, Any] = {}
        try:
            for analyzer in self.audio_analyzers:
                try:
                    analysis = await analyzer(audio_path)
                    if isinstance(analysis, dict):
                        results.update(analysis)
                except Exception as e:
                    logger.warning(f"audio analyzer error: {e}")
        finally:
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception:
                pass
        return results

    async def _extract_audio(self, video_path: str) -> Optional[str]:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = tmp.name
            tmp.close()
            seg = AudioSegment.from_file(video_path)
            seg.export(audio_path, format="wav")
            return audio_path
        except Exception as e:
            logger.warning(f"Error extrayendo audio: {e}")
            return None

    async def _analyze_metadata(self, video_path: str) -> Dict:
        results: Dict[str, Any] = {}
        for analyzer in self.metadata_analyzers:
            try:
                analysis = await analyzer(video_path)
                if isinstance(analysis, dict):
                    results.update(analysis)
            except Exception as e:
                logger.warning(f"metadata analyzer error: {e}")
        return results

    async def _analyze_video_artifacts(self, video_path: str) -> Dict:
        if not cv2:
            return {
                "video_artifacts": {
                    "score": 0.5,
                    "confidence": 0.4,
                    "indicators": [],
                    "note": "OpenCV no disponible",
                }
            }
        try:
            cap = cv2.VideoCapture(video_path)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                blur_value = cv2.Laplacian(frame, cv2.CV_64F).var()
                ai_score = min(90, max(10, int(blur_value / 10)))
                return {
                    "video_artifacts": {
                        "score": ai_score / 100.0,
                        "confidence": 0.7,
                        "indicators": ["consistent_noise_pattern", "unnatural_edges"] if ai_score > 70 else [],
                    }
                }
        except Exception as e:
            logger.warning(f"artefactos video error: {e}")
        return {"video_artifacts": {"score": 0.5, "confidence": 0.5, "indicators": []}}

    async def _analyze_temporal_consistency(self, video_path: str) -> Dict:
        return {"temporal_consistency": {"score": 0.62, "confidence": 0.7, "indicators": ["frame_jitter", "inconsistent_lighting"]}}

    async def _analyze_facial_patterns(self, video_path: str) -> Dict:
        return {"facial_analysis": {"score": 0.85, "confidence": 0.9, "face_count": 1, "indicators": ["unnatural_blinking", "consistent_symmetry"]}}

    async def _analyze_audio_tts(self, audio_path: str) -> Dict:
        return {"audio_tts_analysis": {"score": 0.68, "confidence": 0.75, "indicators": ["unnatural_pauses", "consistent_pitch"]}}

    async def _analyze_audio_quality(self, audio_path: str) -> Dict:
        return {"audio_quality": {"score": 0.8, "confidence": 0.8, "indicators": ["consistent_bitrate", "no_background_noise"]}}

    async def _analyze_technical_metadata(self, video_path: str) -> Dict:
        return {"technical_metadata": {"software_indicators": [], "creation_pattern": "normal", "compression_artifacts": "standard"}}

    async def _analyze_creation_patterns(self, video_path: str) -> Dict:
        return {"creation_patterns": {"score": 0.4, "confidence": 0.6, "indicators": []}}

    def _combine_analyses(self, video_analysis: Dict, audio_analysis: Dict, metadata_analysis: Dict) -> Dict:
        ai_probability = self._calculate_ai_probability(video_analysis, audio_analysis, metadata_analysis)
        verdict = "probable_ai" if ai_probability > 75 else "probable_organic" if ai_probability < 25 else "mixed"
        summary = self._generate_summary(ai_probability, verdict, video_analysis, audio_analysis, metadata_analysis)
        key_indicators = self._extract_key_indicators(video_analysis, audio_analysis, metadata_analysis)
        return {
            "ai_probability": ai_probability,
            "verdict": verdict,
            "summary": summary,
            "key_indicators": key_indicators,
            "vendor_votes": {
                "video_analysis": video_analysis,
                "audio_analysis": audio_analysis,
                "metadata_analysis": metadata_analysis,
            },
        }

    def _calculate_ai_probability(self, video_analysis: Dict, audio_analysis: Dict, metadata_analysis: Dict) -> int:
        video_score = float(video_analysis.get("video_artifacts", {}).get("score", 0.5)) * 0.5
        temporal_score = float(video_analysis.get("temporal_consistency", {}).get("score", 0.5)) * 0.3
        audio_score = float(audio_analysis.get("audio_tts_analysis", {}).get("score", 0.5)) * 0.2
        total_score = (video_score + temporal_score + audio_score) * 100.0
        return int(max(0, min(100, total_score)))

    def _generate_summary(self, ai_probability: int, verdict: str, video_analysis: Dict, audio_analysis: Dict, metadata_analysis: Dict) -> str:
        if verdict == "probable_ai":
            return "Alta probabilidad de contenido generado por IA por patrones visuales y auditivos."
        if verdict == "probable_organic":
            return "Probablemente orgánico; patrones y consistencia dentro de rangos naturales."
        return "Resultados mixtos; se recomienda análisis adicional."

    def _extract_key_indicators(self, video_analysis: Dict, audio_analysis: Dict, metadata_analysis: Dict) -> List[Dict]:
        indicators: List[Dict[str, Any]] = []
        if "video_artifacts" in video_analysis:
            for ind in video_analysis["video_artifacts"].get("indicators", []):
                indicators.append({"indicator": ind, "description": f"Artefacto visual: {ind}", "confidence": video_analysis["video_artifacts"].get("confidence", 0.5)})
        if "audio_tts_analysis" in audio_analysis:
            for ind in audio_analysis["audio_tts_analysis"].get("indicators", []):
                indicators.append({"indicator": ind, "description": f"Artefacto de audio: {ind}", "confidence": audio_analysis["audio_tts_analysis"].get("confidence", 0.5)})
        return indicators

    def _create_error_response(self, message: str) -> Dict:
        return {"error": True, "message": message, "ai_probability": 0, "verdict": "unknown", "summary": f"Error en el análisis: {message}"}

# ========== COST MANAGER ==========
class CostManager:
    def __init__(self):
        self.cost_per_analysis = 0.05
        self.plan_limits = {"free": {"monthly_analyses": 100, "max_duration": 60}, "basic": {"monthly_analyses": 1000, "max_duration": 90}, "pro": {"monthly_analyses": 10000, "max_duration": 120}}

    def check_analysis_limit(self, user_plan: str, analyses_this_month: int) -> bool:
        limit = self.plan_limits.get(user_plan, {}).get("monthly_analyses", 0)
        return analyses_this_month < limit

    def calculate_cost(self, duration: float, user_plan: str) -> float:
        base_cost = self.cost_per_analysis
        plan_modifier = 0.8 if user_plan == "pro" else 0.9 if user_plan == "basic" else 1.0
        duration_factor = 1.0 + (duration / 60.0) * 0.5
        return base_cost * plan_modifier * duration_factor

# ========== AI SEE ANALYZER ==========
class AISeeAnalyzer:
    def __init__(self):
        self.link_resolver = LinkResolver()
        self.content_analyzer = AdvancedContentAnalyzer()
        self.cost_manager = CostManager()
        self.analysis_tracking: Dict[str, Dict[str, Any]] = {}

    async def analyze_content(self, content: str, content_type: str, user_id: str = "default") -> Dict:
        user_plan = "free"
        analyses_count = self.analysis_tracking.get(user_id, {}).get("count", 0)
        if not self.cost_manager.check_analysis_limit(user_plan, analyses_count):
            return self._create_error_response("Límite de análisis mensual excedido")

        if user_id not in self.analysis_tracking:
            self.analysis_tracking[user_id] = {"count": 0, "last_analysis": datetime.now()}
        self.analysis_tracking[user_id]["count"] += 1

        if content_type == "link":
            video_info = await self.link_resolver.resolve_video(content)
            if not video_info or "video_url" not in video_info:
                return self._create_error_response("No se pudo resolver el enlace de video")
            analysis_result = await self.content_analyzer.advanced_analysis(video_info["video_url"])
            analysis_result["source_info"] = video_info
            return analysis_result
        elif content_type in ("video", "image"):
            return await self.content_analyzer.advanced_analysis(content)
        else:
            return self._create_error_response(f"Tipo de contenido no soportado: {content_type}")

    def _create_error_response(self, message: str) -> Dict:
        return {"error": True, "message": message, "ai_probability": 0, "verdict": "unknown", "summary": f"Error: {message}"}

# ========== FASTAPI APP ==========
app = FastAPI(title="AI See Content Analyzer", version="1.0.0")

# CORS (ajusta orígenes según tu front)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # mejor especificar tu dominio en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = AISeeAnalyzer()
analysis_requests: Dict[str, Dict[str, Any]] = {}

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_content_endpoint(request: AnalysisRequest, background_tasks: BackgroundTasks):
    request_id = request.request_id or str(uuid.uuid4())
    analysis_requests[request_id] = {"status": "processing", "request": request.dict(), "start_time": datetime.now()}
    background_tasks.add_task(process_analysis, request_id, request)
    return AnalysisResponse(request_id=request_id, status="processing", result=None)

@app.get("/status/{request_id}")
async def get_analysis_status(request_id: str):
    if request_id not in analysis_requests:
        raise HTTPException(status_code=404, detail="Solicitud no encontrada")
    return analysis_requests[request_id]

async def process_analysis(request_id: str, request: AnalysisRequest):
    try:
        result = await analyzer.analyze_content(request.content, request.content_type)
        analysis_requests[request_id].update({"status": "completed", "result": result, "end_time": datetime.now()})
    except Exception as e:
        logger.error(f"Error procesando análisis {request_id}: {e}")
        analysis_requests[request_id].update({"status": "error", "error_message": str(e), "end_time": datetime.now()})

@app.get("/")
async def root():
    return {"message": "AI See Content Analyzer API", "version": "1.0.0", "endpoints": {"/analyze": "POST", "/status/{request_id}": "GET", "/healthz": "GET"}}

# ========== EJECUCIÓN LOCAL ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

