import asyncio
import logging
from typing import Protocol

import numpy as np
from faster_whisper import WhisperModel

from stt_service.config import AppConfig

logger = logging.getLogger("speech_to_text")


class Transcriber(Protocol):
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        ...


class WhisperTranscriber:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        logger.info(
            "loading whisper model size=%s device=%s compute_type=%s",
            cfg.whisper_model_size,
            cfg.whisper_model_device,
            cfg.whisper_model_compute_type,
        )
        self.model = WhisperModel(
            cfg.whisper_model_size,
            device=cfg.whisper_model_device,
            compute_type=cfg.whisper_model_compute_type,
        )
        logger.info("whisper model loaded")

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        loop = asyncio.get_running_loop()

        def _run() -> tuple[str, float]:
            segments, _ = self.model.transcribe(
                audio,
                language=self.cfg.language,
                vad_filter=self.cfg.whisper_vad_filter,
            )
            items = list(segments)
            text = " ".join(seg.text for seg in items).strip()
            end_sec = float(items[-1].end) if items else (len(audio) / sample_rate)
            return text, end_sec

        return await loop.run_in_executor(None, _run)


class ParakeetTranscriber:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        logger.info(
            "loading parakeet model id=%s device=%s dtype=%s",
            cfg.parakeet_model_id,
            cfg.parakeet_device,
            cfg.parakeet_torch_dtype,
        )

        import torch
        from transformers import pipeline

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(cfg.parakeet_torch_dtype.lower(), torch.float32)

        device = -1
        if cfg.parakeet_device.startswith("cuda"):
            parts = cfg.parakeet_device.split(":", maxsplit=1)
            device = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=cfg.parakeet_model_id,
            trust_remote_code=True,
            device=device,
            torch_dtype=torch_dtype,
        )
        logger.info("parakeet model loaded")

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        loop = asyncio.get_running_loop()

        def _run() -> tuple[str, float]:
            output = self.pipe({"raw": audio, "sampling_rate": sample_rate})
            text = ""
            if isinstance(output, dict):
                text = str(output.get("text", "")).strip()
            elif isinstance(output, str):
                text = output.strip()
            end_sec = len(audio) / sample_rate
            return text, end_sec

        return await loop.run_in_executor(None, _run)
