import asyncio
import logging
import time

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from stt_service.config import AppConfig
from stt_service.security import is_authenticated, is_origin_allowed
from stt_service.session import SessionState
from stt_service.transcribers import Transcriber
from stt_service.ws_utils import (
    normalize_text,
    parse_event_frame,
    sanitize_transcript_text,
    send_event,
)

logger = logging.getLogger("speech_to_text")


def _trim_overlap_prefix(previous_text: str, incoming_text: str) -> str:
    prev = normalize_text(previous_text)
    incoming = sanitize_transcript_text(incoming_text)
    incoming_norm = normalize_text(incoming)
    if not prev or not incoming_norm:
        return incoming

    prev_words = prev.split(" ")
    incoming_words = incoming_norm.split(" ")
    max_overlap = min(len(prev_words), len(incoming_words))
    overlap = 0

    for k in range(max_overlap, 0, -1):
        if prev_words[-k:] == incoming_words[:k]:
            overlap = k
            break

    if overlap == 0:
        return incoming

    trimmed_words = incoming.split(" ")[overlap:]
    return " ".join(trimmed_words).strip()


async def transcribe_and_emit(
    ws: WebSocket,
    state: SessionState,
    cfg: AppConfig,
    transcriber: Transcriber,
    message_type: str,
    reason: str,
    clear_after: bool,
) -> None:
    if state.total_samples < int(cfg.min_audio_sec * cfg.sample_rate):
        if clear_after:
            state.clear()
        return

    audio_full = state.to_audio()
    if audio_full.size == 0:
        if clear_after:
            state.clear()
        return

    overlap_samples = int(cfg.overlap_sec * cfg.sample_rate)
    # Partials run on an overlapping incremental window for responsiveness.
    # Finals should use the full buffered segment for best accuracy.
    if message_type == "final":
        window_start = 0
    else:
        window_start = max(0, state.last_transcribed_samples - overlap_samples)
    audio = audio_full[window_start:]
    if audio.size == 0:
        return

    sem: asyncio.Semaphore = ws.app.state.transcribe_semaphore
    try:
        await asyncio.wait_for(
            sem.acquire(), timeout=cfg.transcribe_acquire_timeout_sec
        )
    except asyncio.TimeoutError:
        logger.warning("transcribe backlog: semaphore wait timed out")
        await send_event(ws, "status", status="busy", reason="backpressure")
        if state.duration_sec >= cfg.max_buffer_sec:
            state.clear()
        return

    try:
        text, model_end_sec = await transcriber.transcribe(audio, cfg.sample_rate)
    except Exception:
        logger.exception("transcription failed")
        await send_event(ws, "error", message="transcription_failed")
        if clear_after:
            state.clear()
        return
    finally:
        sem.release()

    text = sanitize_transcript_text(text)
    if message_type == "final" and window_start > 0:
        text = _trim_overlap_prefix(state.last_final_text, text)
    base_offset_sec = window_start / cfg.sample_rate
    last_end = base_offset_sec + model_end_sec
    signature = (normalize_text(text), round(last_end, 2), message_type)

    if text and signature != state.last_signature:
        state.last_signature = signature
        if message_type == "final":
            state.last_final_text = text
        await send_event(
            ws,
            message_type,
            text=text,
            end=last_end,
            reason=reason,
            window_start_sec=base_offset_sec,
        )

    state.last_transcribed_samples = state.total_samples

    if clear_after:
        state.clear()


async def handle_transcribe_websocket(
    ws: WebSocket,
    cfg: AppConfig,
    transcriber: Transcriber | None,
) -> None:
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

    if transcriber is None:
        await send_event(ws, "error", message="model_unavailable")
        await ws.close(code=1011)
        return

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
                        transcriber,
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

            removed_samples = state.enforce_max_samples(max_buffer_samples)
            if removed_samples > 0:
                logger.warning("buffer trimmed to max size")
                await send_event(
                    ws,
                    "status",
                    status="buffer_trimmed",
                    removed_sec=removed_samples / cfg.sample_rate,
                )

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
                    transcriber,
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
                    transcriber,
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
                    transcriber,
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
