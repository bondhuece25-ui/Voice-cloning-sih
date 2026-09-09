import numpy as np

from app.services.model_service import predict


# Generate 1 second of silence at 16 kHz
audio = np.zeros(
    16000,
    dtype=np.float32
)

print("Running model inference...")

score = predict(audio)

print()
print("================================")
print("MODEL INFERENCE SUCCESS")
print("================================")
print(f"Spoof probability: {score:.6f}")
print("================================")