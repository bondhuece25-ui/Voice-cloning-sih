from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Wav2Vec2Processor

from app.ml.model import VoiceAntiSpoofModel


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "ml" / "models" / "final"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD PROCESSOR
# ============================================================

processor = Wav2Vec2Processor.from_pretrained(
    str(MODEL_DIR)
)


# ============================================================
# CREATE MODEL
# ============================================================

model = VoiceAntiSpoofModel(
    model_name="facebook/wav2vec2-base",
    num_classes=2
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

state_dict = load_file(
    str(MODEL_DIR / "model.safetensors")
)

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)


if missing or unexpected:

    raise RuntimeError(
        f"Model checkpoint mismatch.\n"
        f"Missing keys: {missing}\n"
        f"Unexpected keys: {unexpected}"
    )


# ============================================================
# PREPARE MODEL
# ============================================================

model.to(DEVICE)

model.eval()


# ============================================================
# PREDICTION
# ============================================================

def predict(audio_waveform) -> float:
    """
    Run voice-clone detection on a processed audio waveform.

    Expected input:
        Mono
        16 kHz
        float32 waveform

    Returns:
        float:
            Probability that the audio is SPOOF.
            Value is between 0.0 and 1.0.
    """

    inputs = processor(
        audio_waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_values = inputs.input_values.to(
        DEVICE
    )

    attention_mask = inputs.get(
        "attention_mask"
    )

    if attention_mask is not None:

        attention_mask = attention_mask.to(
            DEVICE
        )


    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    with torch.no_grad():

        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )


    # ========================================================
    # CLASS 1 = SPOOF
    # ========================================================

    spoof_probability = probabilities[
        0, 1
    ].item()


    return spoof_probability