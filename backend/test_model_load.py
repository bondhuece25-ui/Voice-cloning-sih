import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import Wav2Vec2Processor

from app.ml.model import VoiceAntiSpoofModel


MODEL_DIR = Path("ml/models/final")
MODEL_PATH = MODEL_DIR / "model.safetensors"


print("===================================")
print("VOICE CLONE MODEL LOADING TEST")
print("===================================")

print("\n[1/4] Loading processor...")

processor = Wav2Vec2Processor.from_pretrained(
    str(MODEL_DIR)
)

print("✓ Processor loaded")


print("\n[2/4] Creating model architecture...")

model = VoiceAntiSpoofModel(
    model_name="facebook/wav2vec2-base",
    num_classes=2
)

print("✓ Model architecture created")


print("\n[3/4] Loading model weights...")

state_dict = load_file(
    str(MODEL_PATH),
    device="cpu"
)

print(f"✓ Checkpoint loaded")
print(f"  Number of tensors: {len(state_dict)}")


print("\n[4/4] Matching checkpoint with model...")

missing_keys, unexpected_keys = model.load_state_dict(
    state_dict,
    strict=False
)

print("\n-----------------------------------")
print("MODEL COMPATIBILITY RESULT")
print("-----------------------------------")

print(f"Missing keys:    {len(missing_keys)}")
print(f"Unexpected keys: {len(unexpected_keys)}")

if missing_keys:
    print("\nMissing keys:")
    for key in missing_keys:
        print("  ", key)

if unexpected_keys:
    print("\nUnexpected keys:")
    for key in unexpected_keys:
        print("  ", key)

if not missing_keys and not unexpected_keys:
    print("\n🎉 MODEL WEIGHTS MATCH PERFECTLY!")
else:
    print("\n⚠️ MODEL WEIGHTS DO NOT MATCH PERFECTLY.")

model.eval()

print("\nModel is now in evaluation mode.")

print("\n===================================")
print("TEST COMPLETE")
print("===================================")