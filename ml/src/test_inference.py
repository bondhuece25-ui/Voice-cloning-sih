from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Wav2Vec2Processor

from .model import VoiceAntiSpoofModel
from .preprocessing import load_audio, preprocess_audio


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "final"

AUDIO_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "audio"
    / "my_voice.wav"
)


def main() -> None:
    print("Loading trained model...")

    model = VoiceAntiSpoofModel(
        model_name="facebook/wav2vec2-base",
        num_classes=2,
    )

    print("Loading trained weights...")

    # Load one tensor at a time to avoid creating a huge state_dict
    model_parameters = dict(model.named_parameters())

    with safe_open(
        str(MODEL_DIR / "model.safetensors"),
        framework="pt",
        device="cpu",
    ) as f:

        for key in f.keys():
            tensor = f.get_tensor(key)

            if key in model_parameters:
                model_parameters[key].data.copy_(tensor)
            else:
                print(f"Warning: checkpoint key not found in model: {key}")

            del tensor

    print("MODEL FILE LOADED")
    print("MODEL WEIGHTS APPLIED")

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

    prediction = (
        "SPOOF"
        if probabilities[1] > probabilities[0]
        else "BONAFIDE"
    )

    print()
    print("========== RESULT ==========")
    print("Audio:", AUDIO_FILE.name)
    print("Prediction:", prediction)
    print(f"Bonafide probability: {probabilities[0].item():.4f}")
    print(f"Spoof probability: {probabilities[1].item():.4f}")
    print("============================")


if __name__ == "__main__":
    main()