import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from stt_service.config import AppConfig


@dataclass
class SessionState:
    sample_rate: int
    chunks: list[np.ndarray] = field(default_factory=list)
    total_samples: int = 0
    silence_samples: int = 0
    noise_floor: float = 0.001
    last_partial_at: float = field(default_factory=time.monotonic)
    last_signature: Optional[tuple[str, float, str]] = None
    last_transcribed_samples: int = 0
    last_final_text: str = ""

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
        adaptive_threshold = max(
            cfg.silence_rms_floor, self.noise_floor * cfg.silence_multiplier
        )
        is_silence = rms < adaptive_threshold

        if is_silence:
            self.silence_samples += len(chunk)
            self.noise_floor = (
                1 - cfg.noise_alpha
            ) * self.noise_floor + cfg.noise_alpha * max(rms, cfg.min_noise_floor)
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
        self.last_transcribed_samples = 0

    def enforce_max_samples(self, max_samples: int) -> int:
        removed_samples = 0
        while self.total_samples > max_samples and self.chunks:
            removed = self.chunks.pop(0)
            self.total_samples -= len(removed)
            removed_samples += len(removed)
        if removed_samples > 0:
            self.silence_samples = min(self.silence_samples, self.total_samples)
            self.last_transcribed_samples = max(
                0, self.last_transcribed_samples - removed_samples
            )
        return removed_samples
