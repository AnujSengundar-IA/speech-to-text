import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from stt_service.config import AppConfig
from stt_service.stream import handle_transcribe_websocket
from stt_service.transcribers import (
    IndicConformerTranscriber,
    ParakeetTranscriber,
    WhisperTranscriber,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("speech_to_text")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = AppConfig.from_env()
    app.state.config = cfg

    app.state.whisper_transcriber = WhisperTranscriber(cfg)

    try:
        app.state.parakeet_transcriber = ParakeetTranscriber(cfg)
    except Exception:
        app.state.parakeet_transcriber = None
        logger.exception("failed to load parakeet model")

    try:
        app.state.indicconformer_transcriber = IndicConformerTranscriber(cfg)
    except Exception:
        app.state.indicconformer_transcriber = None
        logger.exception("failed to load indicconformer model")

    app.state.transcribe_semaphore = asyncio.Semaphore(cfg.max_concurrent_transcribes)
    try:
        yield
    finally:
        app.state.whisper_transcriber = None
        app.state.parakeet_transcriber = None
        app.state.indicconformer_transcriber = None
        logger.info("shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/transcribe")
async def transcribe_whisper(ws: WebSocket):
    cfg: AppConfig = ws.app.state.config
    await handle_transcribe_websocket(ws, cfg, ws.app.state.whisper_transcriber)


@app.websocket("/ws/transcribe/parakeet")
async def transcribe_parakeet(ws: WebSocket):
    cfg: AppConfig = ws.app.state.config
    await handle_transcribe_websocket(ws, cfg, ws.app.state.parakeet_transcriber)


@app.websocket("/ws/transcribe/indicconformer")
async def transcribe_indicconformer(ws: WebSocket):
    cfg: AppConfig = ws.app.state.config
    await handle_transcribe_websocket(ws, cfg, ws.app.state.indicconformer_transcriber)
