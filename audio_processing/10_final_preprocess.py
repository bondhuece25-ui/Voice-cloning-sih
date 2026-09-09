```python
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import os
import math
import re
import json

from faster_whisper import WhisperModel


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = "audio/sample.wav"

OUTPUT_DIR = "final_output"
CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

TARGET_SR = 16000

# IMPORTANT: 5-second chunks
CHUNK_DURATION = 5


# ============================================================
# SUSPICIOUS TERMS
# ============================================================

SUSPICIOUS_TERMS = [
    "otp",
    "pin",
    "password",
    "bank details",
    "account details",
    "account number",
    "credit card",
    "debit card",
    "card number",
    "cvv",
    "verification code",
    "security code",
    "transaction",
    "upi",
    "upi pin",
    "net banking",
    "bank account",
]


# ============================================================
# SUSPICIOUS TERM DETECTION
# ============================================================

def detect_suspicious_terms(text):

    detected = []

    for term in sorted(
        SUSPICIOUS_TERMS,
        key=len,
        reverse=True
    ):

        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):
            detected.append(term)

    return detected


# ============================================================
# SPOOF DETECTION
# ============================================================

def spoof_detect(audio_file):

    """
    ------------------------------------------------------------
    SPOOF DETECTION INTEGRATION
    ------------------------------------------------------------

    Replace this function with your actual spoof/deepfake
    detection model.

    The model should receive:

        audio_file

    and return something like:

        {
            "status": "completed",
            "is_spoof": True,
            "confidence": 0.92
        }

    or:

        {
            "status": "completed",
            "is_spoof": False,
            "confidence": 0.87
        }

    ------------------------------------------------------------
    TEMPORARY PLACEHOLDER
    ------------------------------------------------------------
    """

    return {
        "status": "not_connected",
        "is_spoof": None,
        "confidence": None
    }


# ============================================================
# LOAD WHISPER
# ============================================================

print()
print("=" * 65)
print("VOXSHIELD - COMPLETE AUDIO PROCESSING PIPELINE")
print("=" * 65)
print()

print("Loading Faster-Whisper model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("✓ Faster-Whisper loaded")
print()


# ============================================================
# 1. LOAD AUDIO
# ============================================================

print("=" * 65)
print("STEP 1 - LOADING AUDIO")
print("=" * 65)

audio, sr = librosa.load(
    INPUT,
    sr=TARGET_SR,
    mono=True
)

original_duration = len(audio) / sr

print(
    f"Original duration: "
    f"{original_duration:.2f} seconds"
)


# ============================================================
# 2. REMOVE SILENCE
# ============================================================

print()
print("=" * 65)
print("STEP 2 - SILENCE REMOVAL")
print("=" * 65)

audio, _ = librosa.effects.trim(
    audio,
    top_db=30
)

trimmed_duration = len(audio) / sr

print(
    f"After silence removal: "
    f"{trimmed_duration:.2f} seconds"
)


# ============================================================
# 3. NOISE REDUCTION
# ============================================================

print()
print("=" * 65)
print("STEP 3 - NOISE REDUCTION")
print("=" * 65)

audio = nr.reduce_noise(
    y=audio,
    sr=sr,
    stationary=False,
    prop_decrease=0.7
)

print("✓ Noise reduction completed")


# ============================================================
# 4. NORMALIZATION
# ============================================================

print()
print("=" * 65)
print("STEP 4 - NORMALIZATION")
print("=" * 65)

audio = librosa.util.normalize(audio)

print("✓ Audio normalized")


# ============================================================
# 5. CREATE 5-SECOND CHUNKS
# ============================================================

print()
print("=" * 65)
print("STEP 5 - CREATING 5-SECOND CHUNKS")
print("=" * 65)

chunk_size = TARGET_SR * CHUNK_DURATION

num_chunks = math.ceil(
    len(audio) / chunk_size
)

print(
    f"Number of chunks: {num_chunks}"
)

print()


# ============================================================
# RESULT STORAGE
# ============================================================

all_results = []

all_transcripts = []

all_suspicious_terms = set()


# ============================================================
# 6. PROCESS EACH CHUNK
# ============================================================

for i in range(num_chunks):

    chunk_number = i + 1

    print()
    print("=" * 65)
    print(
        f"PROCESSING CHUNK {chunk_number}/{num_chunks}"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # Extract chunk
    # --------------------------------------------------------

    start_sample = i * chunk_size
    end_sample = start_sample + chunk_size

    chunk = audio[
        start_sample:end_sample
    ]

    # --------------------------------------------------------
    # Pad final chunk
    # --------------------------------------------------------

    if len(chunk) < chunk_size:

        chunk = np.pad(
            chunk,
            (
                0,
                chunk_size - len(chunk)
            )
        )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    start_time = i * CHUNK_DURATION
    end_time = start_time + CHUNK_DURATION

    # --------------------------------------------------------
    # SAVE PROCESSED CHUNK
    # --------------------------------------------------------

    chunk_file = os.path.join(
        CHUNKS_DIR,
        f"chunk_{chunk_number}.wav"
    )

    sf.write(
        chunk_file,
        chunk,
        TARGET_SR
    )

    print(
        f"✓ Saved: {chunk_file}"
    )

    # ========================================================
    # 7. WHISPER TRANSCRIPTION
    # ========================================================

    print("→ Running Whisper...")

    segments, info = model.transcribe(
        chunk_file,
        language="en",
        condition_on_previous_text=False,
        temperature=0,
        vad_filter=True,
        beam_size=5
    )

    text_parts = []

    for segment in segments:

        text = segment.text.strip()

        if text:
            text_parts.append(text)

    transcript = " ".join(
        text_parts
    )

    if transcript:

        print(
            f"✓ Caption: {transcript}"
        )

    else:

        print(
            "✓ No speech detected"
        )


    # ========================================================
    # 8. SUSPICIOUS TERM DETECTION
    # ========================================================

    suspicious_terms = detect_suspicious_terms(
        transcript
    )

    if suspicious_terms:

        print()
        print("⚠ Suspicious terms:")

        for term in suspicious_terms:

            print(
                f"   - {term}"
            )

            all_suspicious_terms.add(term)

    else:

        print(
            "✓ No suspicious terms"
        )


    # ========================================================
    # 9. SPOOF DETECTION
    # ========================================================

    print()
    print("→ Running spoof detection...")

    spoof_result = spoof_detect(
        chunk_file
    )

    print(
        f"Spoof status: "
        f"{spoof_result['status']}"
    )

    if spoof_result["is_spoof"] is not None:

        print(
            f"Is spoof: "
            f"{spoof_result['is_spoof']}"
        )

        print(
            f"Confidence: "
            f"{spoof_result['confidence']}"
        )


    # ========================================================
    # STORE RESULT
    # ========================================================

    chunk_result = {

        "chunk_number": chunk_number,

        "start_time": start_time,

        "end_time": end_time,

        "file": chunk_file,

        "caption": transcript,

        "suspicious_terms": suspicious_terms,

        "spoof_detection": spoof_result
    }

    all_results.append(
        chunk_result
    )

    all_transcripts.append(
        transcript
    )


# ============================================================
# 10. CREATE FINAL RESULT
# ============================================================

full_transcript = " ".join(
    all_transcripts
)

final_result = {

    "input_file": INPUT,

    "sample_rate": TARGET_SR,

    "chunk_duration": CHUNK_DURATION,

    "number_of_chunks": num_chunks,

    "original_duration": original_duration,

    "duration_after_silence_removal": trimmed_duration,

    "full_transcript": full_transcript,

    "all_suspicious_terms": sorted(
        all_suspicious_terms
    ),

    "chunks": all_results
}


# ============================================================
# 11. SAVE JSON
# ============================================================

json_file = os.path.join(
    OUTPUT_DIR,
    "voxshield_result.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_result,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print()
print("=" * 65)
print("VOXSHIELD PROCESSING COMPLETE")
print("=" * 65)

print()

print(
    f"✓ Original duration: "
    f"{original_duration:.2f}s"
)

print(
    f"✓ Processed duration: "
    f"{trimmed_duration:.2f}s"
)

print(
    f"✓ Number of chunks: "
    f"{num_chunks}"
)

print(
    f"✓ Chunk duration: "
    f"{CHUNK_DURATION}s"
)

print(
    f"✓ Chunks saved in: "
    f"{CHUNKS_DIR}"
)

print(
    f"✓ JSON saved: "
    f"{json_file}"
)

print()

print("=" * 65)
print("FULL TRANSCRIPT")
print("=" * 65)

if full_transcript:

    print(full_transcript)

else:

    print("No speech detected.")


print()

print("=" * 65)
print("SUSPICIOUS TERMS")
print("=" * 65)

if all_suspicious_terms:

    for term in sorted(
        all_suspicious_terms
    ):

        print(
            f"⚠ {term}"
        )

else:

    print(
        "✓ No suspicious terms detected"
    )


print()
print("=" * 65)
print("✓ DONE")
print("=" * 65)
```
