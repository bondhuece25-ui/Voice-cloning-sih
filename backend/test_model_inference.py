import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from transformers import Wav2Vec2Processor

from app.ml.model import VoiceAntiSpoofModel


MODEL_DIR = Path("ml/models/final")
MODEL_PATH = MODEL_DIR / "model.safetensors"

SAMPLE_RATE = 16000


print("===================================")
print("VOICE CLONE INFERENCE TEST")
print("===================================")


# --------------------------------------------------
# 1. Load processor
# --------------------------------------------------

print("\n[1/5] Loading processor...")

processor = Wav2Vec2Processor.from_pretrained(
    str(MODEL_DIR)
)

print("✓ Processor loaded")


# --------------------------------------------------
# 2. Create model
# --------------------------------------------------

print("\n[2/5] Creating model...")

model = VoiceAntiSpoofModel(
    model_name="facebook/wav2vec2-base",
    num_classes=2
)

print("✓ Model architecture created")


# --------------------------------------------------
# 3. Load trained checkpoint
# --------------------------------------------------

print("\n[3/5] Loading trained checkpoint...")

state_dict = load_file(
    str(MODEL_PATH),
    device="cpu"
)

missing_keys, unexpected_keys = model.load_state_dict(
    state_dict,
    strict=False
)

if missing_keys or unexpected_keys:
    print("❌ Model weights do not match.")
    print("Missing:", missing_keys)
    print("Unexpected:", unexpected_keys)
    raise RuntimeError("Checkpoint mismatch")

model.eval()

print("✓ Trained weights loaded")


# --------------------------------------------------
# 4. Create test audio
# --------------------------------------------------

print("\n[4/5] Creating 1-second test audio...")

# 1 second of silence
audio = np.zeros(
    SAMPLE_RATE,
    dtype=np.float32
)

inputs = processor(
    audio,
    sampling_rate=SAMPLE_RATE,
    return_tensors="pt"
)

print("✓ Audio processed")
print("Input shape:", inputs.input_values.shape)


# --------------------------------------------------
# 5. Run inference
# --------------------------------------------------

print("\n[5/5] Running model inference...")

with torch.no_grad():

    outputs = model(
        input_values=inputs.input_values
    )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    spoof_probability = probabilities[0, 1].item()


print("\n===================================")
print("INFERENCE RESULT")
print("===================================")

print("Spoof probability:", spoof_probability)

print("Bonafide probability:", probabilities[0, 0].item())

print("Spoof probability:", probabilities[0, 1].item())


print("\n===================================")

if 0.0 <= spoof_probability <= 1.0:
    print("🎉 REAL MODEL INFERENCE WORKS!")
else:
    print("❌ Invalid probability")

print("===================================")