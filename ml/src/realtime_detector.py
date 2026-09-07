from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Wav2Vec2Processor

from model import VoiceAntiSpoofModel
from preprocessing import preprocess_audio
from risk_engine import RiskEngine


# =========================================================
# Configuration
# =========================================================

MODEL_DIR = Path("../models/final")

SAMPLE_RATE = 16000

# Incoming audio chunk size.
STREAM_CHUNK_DURATION_MS = 500

# Model analyzes the latest 1 second of audio.
INFERENCE_WINDOW_DURATION_MS = 1000

# Keep only enough recent audio for the current window.
# This prevents the buffer from growing indefinitely.
MAX_BUFFER_DURATION_MS = INFERENCE_WINDOW_DURATION_MS

LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70

# Number of recent model predictions used for smoothing.
ROLLING_WINDOW_SIZE = 5


class RealTimeSpoofDetector:
    """
    Streaming voice anti-spoofing detector.

    Audio arrives in approximately 500 ms chunks.
    The model analyzes the latest 1 second.
    Recent predictions are smoothed using RiskEngine.
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:

        self.model_dir = Path(model_dir)
        self.sample_rate = sample_rate

        # -------------------------------------------------
        # Check model directory
        # -------------------------------------------------

        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        model_file = self.model_dir / "model.safetensors"

        if not model_file.is_file():
            raise FileNotFoundError(
                f"Trained model not found: {model_file}"
            )

        # -------------------------------------------------
        # Load processor
        # -------------------------------------------------

        self.processor = Wav2Vec2Processor.from_pretrained(
            self.model_dir
        )

        # -------------------------------------------------
        # Create model architecture
        # -------------------------------------------------

        self.model = VoiceAntiSpoofModel(
            model_name="facebook/wav2vec2-base",
            num_classes=2,
        )

        # -------------------------------------------------
        # Load trained weights
        # -------------------------------------------------

        state_dict = load_file(str(model_file))

        missing, unexpected = self.model.load_state_dict(
            state_dict,
            strict=False,
        )

        if missing or unexpected:
            raise RuntimeError(
                "Saved model weights do not match the "
                "anti-spoofing model architecture."
            )

        # -------------------------------------------------
        # Select device
        # -------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        # -------------------------------------------------
        # Audio buffer
        # -------------------------------------------------

        self.audio_buffer = np.array(
            [],
            dtype=np.float32,
        )

        # Maximum number of samples allowed in buffer.
        self.max_buffer_samples = round(
            self.sample_rate
            * MAX_BUFFER_DURATION_MS
            / 1000
        )

        # -------------------------------------------------
        # Risk engine
        # -------------------------------------------------

        self.risk_engine = RiskEngine(
            low_threshold=LOW_RISK_THRESHOLD,
            high_threshold=HIGH_RISK_THRESHOLD,
            window_size=ROLLING_WINDOW_SIZE,
        )

    def add_audio_chunk(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float | str | None]:
        """
        Add one incoming audio chunk.

        Expected chunk duration:
            approximately 500 ms.

        Once one second of audio is available,
        the latest 1-second window is analyzed.
        """

        # -------------------------------------------------
        # Validate audio
        # -------------------------------------------------

        chunk = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if chunk.ndim != 1:
            raise ValueError(
                "Audio chunk must be one-dimensional."
            )

        if chunk.size == 0:
            raise ValueError(
                "Audio chunk is empty."
            )

        # -------------------------------------------------
        # Preprocess audio
        # -------------------------------------------------

        chunk, _ = preprocess_audio(
            chunk,
            sample_rate,
            self.sample_rate,
        )

        # -------------------------------------------------
        # Add chunk to buffer
        # -------------------------------------------------

        self.audio_buffer = np.concatenate(
            [
                self.audio_buffer,
                chunk,
            ]
        )

        # -------------------------------------------------
        # Prevent unbounded buffer growth
        # -------------------------------------------------

        if len(self.audio_buffer) > self.max_buffer_samples:
            self.audio_buffer = self.audio_buffer[
                -self.max_buffer_samples:
            ]

        # -------------------------------------------------
        # Required model window
        # -------------------------------------------------

        inference_size = round(
            self.sample_rate
            * INFERENCE_WINDOW_DURATION_MS
            / 1000
        )

        # -------------------------------------------------
        # Wait for one second of audio
        # -------------------------------------------------

        if len(self.audio_buffer) < inference_size:
            return {
                "prediction": None,
                "spoof_probability": None,
                "rolling_risk": None,
                "decision": "WAITING_FOR_AUDIO",
            }

        # -------------------------------------------------
        # Select latest 1-second window
        # -------------------------------------------------

        inference_audio = self.audio_buffer[
            -inference_size:
        ]

        # -------------------------------------------------
        # Run model
        # -------------------------------------------------

        spoof_probability = (
            self._predict_spoof_probability(
                inference_audio
            )
        )

        # -------------------------------------------------
        # Update rolling risk
        # -------------------------------------------------

        risk_result = self.risk_engine.update(
            spoof_probability
        )

        rolling_risk = float(
            risk_result["rolling_risk"]
        )

        decision = str(
            risk_result["decision"]
        )

        # -------------------------------------------------
        # Current window classification
        # -------------------------------------------------

        prediction = (
            "SPOOF"
            if spoof_probability >= HIGH_RISK_THRESHOLD
            else "BONAFIDE"
        )

        return {
            "prediction": prediction,
            "spoof_probability": spoof_probability,
            "rolling_risk": rolling_risk,
            "decision": decision,
        }

    def _predict_spoof_probability(
        self,
        waveform: np.ndarray,
    ) -> float:
        """
        Perform one Wav2Vec2 inference on a 1-second window.
        """

        inputs = self.processor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        # Class 1 = SPOOF
        return float(
            probabilities[1].item()
        )