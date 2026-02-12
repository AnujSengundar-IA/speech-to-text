import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("speech_to_text")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str) -> list[str]:
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class AppConfig:
    model_size: str = "tiny.en"
    model_device: str = "cpu"
    model_compute_type: str = "int8"
    language: str = "en"
    sample_rate: int = 16000
    min_audio_sec: float = 0.5
    silence_sec: float = 1.0
    max_segment_sec: float = 10.0
    max_buffer_sec: float = 15.0
    partial_interval_sec: float = 1.0
    silence_multiplier: float = 2.5
    silence_rms_floor: float = 0.003
    noise_alpha: float = 0.1
    min_noise_floor: float = 0.0005
    vad_filter: bool = True
    client_timeout_sec: float = 20.0
    max_concurrent_transcribes: int = 2
    transcribe_acquire_timeout_sec: float = 5.0
    allowed_origins: list[str] = field(default_factory=list)
    ws_auth_token: str = ""

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            model_size=os.getenv("WHISPER_MODEL_SIZE", "tiny.en"),
            model_device=os.getenv("WHISPER_MODEL_DEVICE", "cpu"),
            model_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            language=os.getenv("WHISPER_LANGUAGE", "en"),
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            min_audio_sec=float(os.getenv("MIN_AUDIO_SEC", "0.5")),
            silence_sec=float(os.getenv("SILENCE_SEC", "1.0")),
            max_segment_sec=float(os.getenv("MAX_SEGMENT_SEC", "10.0")),
            max_buffer_sec=float(os.getenv("MAX_BUFFER_SEC", "15.0")),
            partial_interval_sec=float(os.getenv("PARTIAL_INTERVAL_SEC", "1.0")),
            silence_multiplier=float(os.getenv("SILENCE_MULTIPLIER", "2.5")),
            silence_rms_floor=float(os.getenv("SILENCE_RMS_FLOOR", "0.003")),
            noise_alpha=float(os.getenv("NOISE_ALPHA", "0.1")),
            min_noise_floor=float(os.getenv("MIN_NOISE_FLOOR", "0.0005")),
            vad_filter=env_bool("WHISPER_VAD_FILTER", True),
            client_timeout_sec=float(os.getenv("CLIENT_TIMEOUT_SEC", "20")),
            max_concurrent_transcribes=int(os.getenv("MAX_CONCURRENT_TRANSCRIBES", "2")),
            transcribe_acquire_timeout_sec=float(
                os.getenv("TRANSCRIBE_ACQUIRE_TIMEOUT_SEC", "5")
            ),
            allowed_origins=env_list("WS_ALLOWED_ORIGINS"),
            ws_auth_token=os.getenv("WS_AUTH_TOKEN", ""),
        )


@dataclass
class SessionState:
    sample_rate: int
    chunks: list[np.ndarray] = field(default_factory=list)
    total_samples: int = 0
    silence_samples: int = 0
    noise_floor: float = 0.001
    last_partial_at: float = field(default_factory=time.monotonic)
    last_signature: Optional[tuple[str, float, str]] = None

    @property
    def duration_sec(self) -> float:
        return self.total_samples / self.sample_rate

    @property
    def silence_duration_sec(self) -> float:
        return self.silence_samples / self.sample_rate

    def add_chunk(self, chunk: np.ndarray, cfg: AppConfig) -> tuple[float, float]:
        self.chunks.append(chunk)
        self.total_samples += len(chunk)

        rms = float(np.sqrt(np.mean(np.square(chunk))))
        adaptive_threshold = max(cfg.silence_rms_floor, self.noise_floor * cfg.silence_multiplier)
        is_silence = rms < adaptive_threshold

        if is_silence:
            self.silence_samples += len(chunk)
            self.noise_floor = (1 - cfg.noise_alpha) * self.noise_floor + cfg.noise_alpha * max(
                rms, cfg.min_noise_floor
            )
        else:
            self.silence_samples = 0

        return rms, adaptive_threshold

    def to_audio(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()
        self.total_samples = 0
        self.silence_samples = 0

    def enforce_max_samples(self, max_samples: int) -> bool:
        trimmed = False
        while self.total_samples > max_samples and self.chunks:
            removed = self.chunks.pop(0)
            self.total_samples -= len(removed)
            trimmed = True
        if trimmed:
            self.silence_samples = min(self.silence_samples, self.total_samples)
        return trimmed


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip().lower()
    return cleaned


async def send_event(ws: WebSocket, event_type: str, **payload: object) -> None:
    await ws.send_json({"type": event_type, **payload})


def parse_event_frame(text_frame: str) -> Optional[str]:
    try:
        data = json.loads(text_frame)
    except JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    event = data.get("event")
    if isinstance(event, str):
        return event
    return None


async def run_model_transcription(
    model: WhisperModel,
    audio: np.ndarray,
    cfg: AppConfig,
) -> tuple[list[object], object]:
    loop = asyncio.get_running_loop()

    def _run() -> tuple[list[object], object]:
        segments, info = model.transcribe(
            audio,
            language=cfg.language,
            vad_filter=cfg.vad_filter,
        )
        return list(segments), info

    return await loop.run_in_executor(None, _run)


async def transcribe_and_emit(
    ws: WebSocket,
    state: SessionState,
    cfg: AppConfig,
    message_type: str,
    reason: str,
    clear_after: bool,
) -> None:
    if state.total_samples < int(cfg.min_audio_sec * cfg.sample_rate):
        if clear_after:
            state.clear()
        return

    audio = state.to_audio()
    if audio.size == 0:
        if clear_after:
            state.clear()
        return

    sem: asyncio.Semaphore = ws.app.state.transcribe_semaphore
    try:
        await asyncio.wait_for(sem.acquire(), timeout=cfg.transcribe_acquire_timeout_sec)
    except asyncio.TimeoutError:
        logger.warning("transcribe backlog: semaphore wait timed out")
        await send_event(ws, "status", status="busy", reason="backpressure")
        if state.duration_sec >= cfg.max_buffer_sec:
            state.clear()
        return

    try:
        segments, _ = await run_model_transcription(ws.app.state.model, audio, cfg)
    except Exception:
        logger.exception("transcription failed")
        await send_event(ws, "error", message="transcription_failed")
        if clear_after:
            state.clear()
        return
    finally:
        sem.release()

    text = " ".join(seg.text for seg in segments).strip()
    last_end = float(segments[-1].end) if segments else 0.0
    signature = (normalize_text(text), round(last_end, 2), message_type)

    if text and signature != state.last_signature:
        state.last_signature = signature
        await send_event(
            ws,
            message_type,
            text=text,
            end=last_end,
            reason=reason,
        )

    if clear_after:
        state.clear()


def is_origin_allowed(ws: WebSocket, cfg: AppConfig) -> bool:
    if not cfg.allowed_origins:
        return True
    origin = ws.headers.get("origin", "")
    return origin in cfg.allowed_origins


def is_authenticated(ws: WebSocket, cfg: AppConfig) -> bool:
    if not cfg.ws_auth_token:
        return True
    token = ws.query_params.get("token") or ws.headers.get("x-api-key", "")
    return token == cfg.ws_auth_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = AppConfig.from_env()
    app.state.config = cfg
    logger.info(
        "loading model size=%s device=%s compute_type=%s",
        cfg.model_size,
        cfg.model_device,
        cfg.model_compute_type,
    )
    app.state.model = WhisperModel(
        cfg.model_size,
        device=cfg.model_device,
        compute_type=cfg.model_compute_type,
    )
    app.state.transcribe_semaphore = asyncio.Semaphore(cfg.max_concurrent_transcribes)
    logger.info("model loaded")
    try:
        yield
    finally:
        app.state.model = None
        logger.info("shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/transcribe")
async def transcribe(ws: WebSocket):
    cfg: AppConfig = ws.app.state.config

    if not is_origin_allowed(ws, cfg):
        logger.warning("websocket rejected due to origin: %s", ws.headers.get("origin"))
        await ws.close(code=1008)
        return

    if not is_authenticated(ws, cfg):
        logger.warning("websocket rejected due to missing/invalid token")
        await ws.close(code=1008)
        return

    await ws.accept()
    await send_event(ws, "status", status="connected")
    logger.info("websocket accepted")

    state = SessionState(sample_rate=cfg.sample_rate)
    max_buffer_samples = int(cfg.max_buffer_sec * cfg.sample_rate)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=cfg.client_timeout_sec)
            except asyncio.TimeoutError:
                logger.warning("closing stale websocket due to inactivity")
                await send_event(ws, "error", message="client_timeout")
                await ws.close(code=1001)
                break

            if msg.get("type") == "websocket.disconnect":
                logger.info("disconnect frame received")
                break

            if msg.get("type") != "websocket.receive":
                continue

            text_frame = msg.get("text")
            if text_frame is not None:
                event = parse_event_frame(text_frame)
                if event is None:
                    logger.warning("invalid text frame ignored")
                    continue
                if event == "ping":
                    await send_event(ws, "status", status="pong")
                    continue
                if event == "stop":
                    await transcribe_and_emit(
                        ws,
                        state,
                        cfg,
                        message_type="final",
                        reason="stop",
                        clear_after=True,
                    )
                    break
                logger.warning("unknown event ignored: %s", event)
                continue

            frame_bytes = msg.get("bytes")
            if frame_bytes is None:
                continue

            chunk = (
                np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
            if chunk.size == 0:
                continue

            rms, threshold = state.add_chunk(chunk, cfg)
            logger.debug(
                "audio chunk rms=%.5f threshold=%.5f dur=%.2fs silence=%.2fs",
                rms,
                threshold,
                state.duration_sec,
                state.silence_duration_sec,
            )

            if state.enforce_max_samples(max_buffer_samples):
                logger.warning("buffer trimmed to max size")
                await send_event(ws, "status", status="buffer_trimmed")

            now = time.monotonic()
            should_partial = (
                state.duration_sec >= cfg.min_audio_sec
                and state.silence_duration_sec < cfg.silence_sec
                and (now - state.last_partial_at) >= cfg.partial_interval_sec
            )
            if should_partial:
                await transcribe_and_emit(
                    ws,
                    state,
                    cfg,
                    message_type="partial",
                    reason="interval",
                    clear_after=False,
                )
                state.last_partial_at = now

            should_finalize = (
                state.silence_duration_sec >= cfg.silence_sec
                or state.duration_sec >= cfg.max_segment_sec
            ) and state.duration_sec >= cfg.min_audio_sec
            if should_finalize:
                reason = (
                    "silence"
                    if state.silence_duration_sec >= cfg.silence_sec
                    else "max_segment"
                )
                await transcribe_and_emit(
                    ws,
                    state,
                    cfg,
                    message_type="final",
                    reason=reason,
                    clear_after=True,
                )
                state.last_partial_at = time.monotonic()

    except WebSocketDisconnect:
        logger.info("client disconnected")
    except Exception:
        logger.exception("unexpected websocket error")
        try:
            await send_event(ws, "error", message="internal_error")
        except Exception:
            logger.exception("failed to send internal_error event")
    finally:
        if state.total_samples > 0:
            try:
                await transcribe_and_emit(
                    ws,
                    state,
                    cfg,
                    message_type="final",
                    reason="final_flush",
                    clear_after=True,
                )
            except Exception:
                logger.exception("final flush failed")

        try:
            await ws.close()
        except RuntimeError as exc:
            logger.warning("websocket close runtime error: %s", exc)
        except Exception:
            logger.exception("websocket close failed")

        logger.info("websocket closed")
