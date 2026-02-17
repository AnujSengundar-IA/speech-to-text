import os
from dataclasses import dataclass, field


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
    whisper_model_size: str = "small.en"
    whisper_model_device: str = "cpu"
    whisper_model_compute_type: str = "int8"

    parakeet_model_id: str = "nvidia/parakeet-ctc-0.6b"
    parakeet_device: str = "cpu"
    parakeet_torch_dtype: str = "float32"
    indicconformer_checkpoint_path: str = (
        "ai4bharat/indic-conformer-600m-multilingual"
    )
    indicconformer_device: str = "cpu"
    indicconformer_decoder: str = "ctc"
    indicconformer_default_language_id: str = "hi"
    indicconformer_batch_size: int = 1

    language: str = "en"
    sample_rate: int = 16000
    min_audio_sec: float = 0.5
    silence_sec: float = 1.0
    max_segment_sec: float = 10.0
    max_buffer_sec: float = 15.0
    partial_interval_sec: float = 1.0
    overlap_sec: float = 1.0
    silence_multiplier: float = 2.5
    silence_rms_floor: float = 0.003
    noise_alpha: float = 0.1
    min_noise_floor: float = 0.0005

    whisper_vad_filter: bool = True
    client_timeout_sec: float = 20.0
    max_concurrent_transcribes: int = 2
    transcribe_acquire_timeout_sec: float = 5.0
    allowed_origins: list[str] = field(default_factory=list)
    ws_auth_token: str = ""

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small.en"),
            whisper_model_device=os.getenv("WHISPER_MODEL_DEVICE", "cpu"),
            whisper_model_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            parakeet_model_id=os.getenv(
                "PARAKEET_MODEL_ID", "nvidia/parakeet-ctc-0.6b"
            ),
            parakeet_device=os.getenv("PARAKEET_DEVICE", "cpu"),
            parakeet_torch_dtype=os.getenv("PARAKEET_TORCH_DTYPE", "float32"),
            indicconformer_checkpoint_path=os.getenv(
                "INDICCONFORMER_CHECKPOINT_PATH",
                "ai4bharat/indic-conformer-600m-multilingual",
            ),
            indicconformer_device=os.getenv("INDICCONFORMER_DEVICE", "cpu"),
            indicconformer_decoder=os.getenv("INDICCONFORMER_DECODER", "ctc"),
            indicconformer_default_language_id=os.getenv(
                "INDICCONFORMER_DEFAULT_LANGUAGE_ID", "hi"
            ),
            indicconformer_batch_size=int(
                os.getenv("INDICCONFORMER_BATCH_SIZE", "1")
            ),
            language=os.getenv("WHISPER_LANGUAGE", "en"),
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            min_audio_sec=float(os.getenv("MIN_AUDIO_SEC", "0.5")),
            silence_sec=float(os.getenv("SILENCE_SEC", "1.0")),
            max_segment_sec=float(os.getenv("MAX_SEGMENT_SEC", "10.0")),
            max_buffer_sec=float(os.getenv("MAX_BUFFER_SEC", "15.0")),
            partial_interval_sec=float(os.getenv("PARTIAL_INTERVAL_SEC", "1.0")),
            overlap_sec=float(os.getenv("OVERLAP_SEC", "1.0")),
            silence_multiplier=float(os.getenv("SILENCE_MULTIPLIER", "2.5")),
            silence_rms_floor=float(os.getenv("SILENCE_RMS_FLOOR", "0.003")),
            noise_alpha=float(os.getenv("NOISE_ALPHA", "0.1")),
            min_noise_floor=float(os.getenv("MIN_NOISE_FLOOR", "0.0005")),
            whisper_vad_filter=env_bool("WHISPER_VAD_FILTER", True),
            client_timeout_sec=float(os.getenv("CLIENT_TIMEOUT_SEC", "20")),
            max_concurrent_transcribes=int(
                os.getenv("MAX_CONCURRENT_TRANSCRIBES", "2")
            ),
            transcribe_acquire_timeout_sec=float(
                os.getenv("TRANSCRIBE_ACQUIRE_TIMEOUT_SEC", "5")
            ),
            allowed_origins=env_list("WS_ALLOWED_ORIGINS"),
            ws_auth_token=os.getenv("WS_AUTH_TOKEN", ""),
        )
