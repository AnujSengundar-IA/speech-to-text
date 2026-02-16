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
        self._hf = HFASRTranscriber(
            model_id=cfg.parakeet_model_id,
            device_name=cfg.parakeet_device,
            dtype_name=cfg.parakeet_torch_dtype,
            label="parakeet",
        )

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> tuple[str, float]:
        return await self._hf.transcribe(audio, sample_rate)


class HFASRTranscriber:
    def __init__(
        self,
        model_id: str,
        device_name: str,
        dtype_name: str,
        label: str,
    ):
        logger.info(
            "loading %s model id=%s device=%s dtype=%s",
            label,
            model_id,
            device_name,
            dtype_name,
        )

        import torch
        from transformers import pipeline

        # TDT Parakeet checkpoints are distributed as NeMo archives (.nemo)
        # and are not loadable through the generic HF ASR pipeline.
        if label == "parakeet" and "parakeet-tdt" in model_id.lower():
            raise ValueError(
                "Parakeet TDT checkpoints are not supported by the transformers ASR "
                "pipeline in this project. Use a Parakeet CTC model id "
                "(for example: nvidia/parakeet-ctc-0.6b) or add a NeMo backend."
            )

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype_name.lower(), torch.float32)

        device = -1
        if device_name.startswith("cuda"):
            parts = device_name.split(":", maxsplit=1)
            device = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            trust_remote_code=True,
            device=device,
            dtype=torch_dtype,
        )
        logger.info("%s model loaded", label)

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
