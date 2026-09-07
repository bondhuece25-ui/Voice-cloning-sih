from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Wav2Vec2Processor

from model import VoiceAntiSpoofModel
from preprocessing import load_audio, preprocess_audio


MODEL_DIR = Path("../models/final")
AUDIO_FILE = Path("../data/raw/audio/spoof_0000.wav")

def main() -> None:
    print("Loading trained model...")

    model = VoiceAntiSpoofModel(
        model_name="facebook/wav2vec2-base",
        num_classes=2,
    )

    state_dict = load_file(str(MODEL_DIR / "model.safetensors"))
    model.load_state_dict(state_dict)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

    waveform, sample_rate = load_audio(AUDIO_FILE)
    waveform, sample_rate = preprocess_audio(waveform, sample_rate)

    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    prediction = "SPOOF" if probabilities[1] > probabilities[0] else "BONAFIDE"

    print("Audio:", AUDIO_FILE.name)
    print("Prediction:", prediction)
    print(f"Bonafide probability: {probabilities[0].item():.4f}")
    print(f"Spoof probability: {probabilities[1].item():.4f}")


if __name__ == "__main__":
    main()