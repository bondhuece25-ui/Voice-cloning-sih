import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import os
import math

INPUT = "audio/sample.wav"

OUTPUT_DIR = "final_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
CHUNK_DURATION = 4

# --------------------------------------------------
# 1. Load audio
# --------------------------------------------------

audio, sr = librosa.load(
    INPUT,
    sr=TARGET_SR,
    mono=True
)

print("Original duration:", len(audio) / sr, "seconds")


# --------------------------------------------------
# 2. Remove silence
# --------------------------------------------------

audio, _ = librosa.effects.trim(
    audio,
    top_db=30
)

print("After silence removal:", len(audio) / sr, "seconds")


# --------------------------------------------------
# 3. Noise reduction
# --------------------------------------------------

audio = nr.reduce_noise(
    y=audio,
    sr=sr,
    stationary=False,
    prop_decrease=0.7
)

print("Noise reduction completed")


# --------------------------------------------------
# 4. Normalize
# --------------------------------------------------

audio = librosa.util.normalize(audio)


# --------------------------------------------------
# 5. Split into 4-second chunks
# --------------------------------------------------

chunk_size = TARGET_SR * CHUNK_DURATION

num_chunks = math.ceil(len(audio) / chunk_size)

print("Number of chunks:", num_chunks)


for i in range(num_chunks):

    start = i * chunk_size
    end = start + chunk_size

    chunk = audio[start:end]

    # Pad final chunk
    if len(chunk) < chunk_size:

        chunk = np.pad(
            chunk,
            (0, chunk_size - len(chunk))
        )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"chunk_{i+1}.wav"
    )

    sf.write(
        output_file,
        chunk,
        TARGET_SR
    )

    print("Saved:", output_file)


print("\nProcessing complete!")