from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import INFERENCE_CHUNK_DURATION_MS, SAMPLE_RATE


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV or FLAC file as a float32 waveform and sample rate."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError) as error:
        raise ValueError(f"Could not read audio file: {audio_path}") from error

    if waveform.size == 0 or waveform.shape[0] == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    mono_waveform = waveform.mean(axis=1, dtype=np.float32)
    return mono_waveform, int(sample_rate)


def preprocess_audio(
    waveform: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Convert audio to mono float32, resample it, and normalize it safely."""
    audio = np.asarray(waveform)
    if audio.size == 0:
        raise ValueError("Audio waveform is empty")
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive")

    if audio.ndim == 2:
        audio = audio.mean(axis=1 if audio.shape[1] <= audio.shape[0] else 0)
    elif audio.ndim != 1:
        raise ValueError("Audio waveform must be one- or two-dimensional")

    audio = audio.astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        raise ValueError("Audio waveform contains non-finite values")

    if sample_rate != target_sample_rate:
        common_divisor = np.gcd(sample_rate, target_sample_rate)
        audio = resample_poly(
            audio,
            target_sample_rate // common_divisor,
            sample_rate // common_divisor,
        ).astype(np.float32)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32, copy=False), target_sample_rate


def split_into_chunks(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    chunk_duration_ms: int = INFERENCE_CHUNK_DURATION_MS,
) -> list[np.ndarray]:
    """Split a waveform into chunks without padding the final chunk."""
    audio = np.asarray(waveform)
    if audio.size == 0:
        raise ValueError("Audio waveform is empty")
    if audio.ndim != 1:
        raise ValueError("Audio waveform must be one-dimensional")
    if sample_rate <= 0 or chunk_duration_ms <= 0:
        raise ValueError("Sample rate and chunk duration must be positive")

    chunk_size = round(sample_rate * chunk_duration_ms / 1000)
    if chunk_size <= 0:
        raise ValueError("Chunk duration is too short for the sample rate")

    return [audio[start : start + chunk_size] for start in range(0, len(audio), chunk_size)]