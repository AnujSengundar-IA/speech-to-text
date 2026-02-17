import asyncio
import importlib.util
import logging
import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Protocol

import numpy as np
from faster_whisper import WhisperModel

from stt_service.config import AppConfig

logger = logging.getLogger("speech_to_text")


class Transcriber(Protocol):
    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language_id: str | None = None,
    ) -> tuple[str, float]:
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

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language_id: str | None = None,
    ) -> tuple[str, float]:
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

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language_id: str | None = None,
    ) -> tuple[str, float]:
        return await self._hf.transcribe(audio, sample_rate, language_id=language_id)


class IndicConformerTranscriber:
    def __init__(self, cfg: AppConfig):
        model_source = cfg.indicconformer_checkpoint_path.strip()
        if not model_source:
            raise ValueError("INDICCONFORMER_CHECKPOINT_PATH is required")

        self.cfg = cfg
        self.decoder = self._normalize_decoder(cfg.indicconformer_decoder)
        self.supported_languages: set[str] | None = None

        if model_source.lower().endswith(".nemo"):
            self._load_nemo(model_source)
        else:
            self._load_hf_repo(model_source)

    def _normalize_decoder(self, decoder: str) -> str:
        normalized = decoder.strip().lower()
        if normalized not in {"ctc", "rnnt"}:
            raise ValueError(
                "INDICCONFORMER_DECODER must be either 'ctc' or 'rnnt', got "
                f"{decoder!r}"
            )
        return normalized

    def _load_nemo(self, checkpoint_path: str) -> None:
        logger.info(
            "loading indicconformer nemo checkpoint=%s device=%s decoder=%s",
            checkpoint_path,
            self.cfg.indicconformer_device,
            self.decoder,
        )

        import torch
        import nemo.collections.asr as nemo_asr

        self.mode = "nemo"
        self.device = torch.device(self.cfg.indicconformer_device)
        self.model = nemo_asr.models.EncDecCTCModel.restore_from(
            restore_path=checkpoint_path
        )
        self.model.freeze()
        self.model = self.model.to(self.device)
        self.model.cur_decoder = self.decoder
        logger.info("indicconformer nemo model loaded")

    def _resolve_hf_snapshot_dir(self, model_source: str) -> str:
        expanded = os.path.expanduser(model_source)
        path = Path(expanded)
        if path.exists():
            if not path.is_dir():
                raise ValueError(
                    "INDICCONFORMER_CHECKPOINT_PATH must be a directory, .nemo file, "
                    f"or HF repo id. Got file path: {expanded}"
                )
            return str(path)

        from huggingface_hub import snapshot_download

        logger.info("downloading indicconformer snapshot for repo id=%s", model_source)
        return snapshot_download(repo_id=model_source)

    def _load_hf_repo(self, model_source: str) -> None:
        model_dir = self._resolve_hf_snapshot_dir(model_source)
        module_candidates: list[str] = []
        if os.path.exists(os.path.join(model_dir, "assets", "encoder.onnx")):
            module_candidates.extend(
                [
                    os.path.join(model_dir, "model_onnx.py"),
                    os.path.join(model_dir, "model_onnx_1b_batched_rnnt.py"),
                ]
            )
        module_candidates.append(os.path.join(model_dir, "model_ts.py"))

        module_path = next((p for p in module_candidates if os.path.exists(p)), None)
        if module_path is None:
            raise ValueError(
                "Could not find a supported IndicConformer model entrypoint in "
                f"{model_dir}. Expected one of: model_onnx.py, "
                "model_onnx_1b_batched_rnnt.py, model_ts.py"
            )

        logger.info(
            "loading indicconformer hf-ts model_dir=%s device=%s decoder=%s",
            model_dir,
            self.cfg.indicconformer_device,
            self.decoder,
        )

        module_name = (
            "indicconformer_model_"
            + str(abs(hash(os.path.abspath(module_path))))
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load model module from {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        config_cls = getattr(module, "IndicASRConfig")
        model_cls = getattr(module, "IndicASRModel")

        self.mode = "hf_ts"
        self.model = model_cls(
            config_cls(
                ts_folder=model_dir,
                device=self.cfg.indicconformer_device,
            )
        )
        lang_masks = getattr(self.model, "language_masks", None)
        if isinstance(lang_masks, dict):
            self.supported_languages = set(lang_masks.keys())
        logger.info("indicconformer hf-ts model loaded")

    def _resolve_language_id(self, language_id: str | None) -> str:
        fallback = self.cfg.indicconformer_default_language_id.strip() or "hi"
        candidate = (language_id or "").strip() or fallback
        if self.supported_languages and candidate not in self.supported_languages:
            logger.warning(
                "unsupported language_id=%s for indicconformer; falling back to %s",
                candidate,
                fallback,
            )
            return fallback
        return candidate

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language_id: str | None = None,
    ) -> tuple[str, float]:
        loop = asyncio.get_running_loop()

        def _run() -> tuple[str, float]:
            lang = self._resolve_language_id(language_id)
            end_sec = len(audio) / sample_rate

            if self.mode == "nemo":
                fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="indicconformer_")
                os.close(fd)
                try:
                    _write_wav(tmp_path, audio, sample_rate)
                    kwargs = {
                        "batch_size": max(1, self.cfg.indicconformer_batch_size),
                        "language_id": lang,
                    }
                    if self.decoder == "ctc":
                        kwargs["logprobs"] = False
                    result = self.model.transcribe([tmp_path], **kwargs)
                    text = str(result[0]).strip() if result else ""
                    return text, end_sec
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        logger.warning("failed to remove temp wav file: %s", tmp_path)

            import torch

            wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            output = self.model(wav, lang, decoding=self.decoder)
            if isinstance(output, tuple):
                text = str(output[0]).strip()
            else:
                text = str(output).strip()
            return text, end_sec

        return await loop.run_in_executor(None, _run)


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

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language_id: str | None = None,
    ) -> tuple[str, float]:
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


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())
