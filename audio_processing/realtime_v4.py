from faster_whisper import WhisperModel
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr

import threading
import queue
import tempfile
import os
import re
import json
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1

CHUNK_DURATION = 5
OVERLAP_DURATION = 1

MODEL_SIZE = "base"

OUTPUT_JSON = "voxshield_result.json"

PROCESSED_DIR = "processed_chunks"

os.makedirs(PROCESSED_DIR, exist_ok=True)


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
# QUEUE
# ============================================================

audio_queue = queue.Queue()

stop_event = threading.Event()


# ============================================================
# STORE CONVERSATION
# ============================================================

conversation = []

all_suspicious_terms = set()


# ============================================================
# PREVIOUS PROCESSED AUDIO FOR OVERLAP
# ============================================================

previous_audio = None


# ============================================================
# SUSPICIOUS TERM DETECTION
# ============================================================

def detect_suspicious_terms(text):

    detected = []

    for term in sorted(SUSPICIOUS_TERMS, key=len, reverse=True):

        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            detected.append(term)

    return detected


# ============================================================
# PREPROCESS AUDIO
# ============================================================

def preprocess_audio(audio):

    """
    Applies the same preprocessing used in preprocess.py:

    1. Silence removal
    2. Noise reduction
    3. Normalization

    The processed result is then padded/cropped
    to exactly 5 seconds.
    """

    # --------------------------------------------------------
    # Make sure audio is 1D
    # --------------------------------------------------------

    audio = audio.flatten()

    # --------------------------------------------------------
    # 1. REMOVE SILENCE
    # --------------------------------------------------------

    processed_audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    # --------------------------------------------------------
    # If nothing meaningful remains
    # --------------------------------------------------------

    if len(processed_audio) == 0:

        return np.zeros(
            int(CHUNK_DURATION * SAMPLE_RATE),
            dtype=np.float32
        )

    # --------------------------------------------------------
    # 2. NOISE REDUCTION
    # --------------------------------------------------------

    processed_audio = nr.reduce_noise(
        y=processed_audio,
        sr=SAMPLE_RATE,
        stationary=False,
        prop_decrease=0.7
    )

    # --------------------------------------------------------
    # 3. NORMALIZE
    # --------------------------------------------------------

    if np.max(np.abs(processed_audio)) > 0:

        processed_audio = librosa.util.normalize(
            processed_audio
        )

    # --------------------------------------------------------
    # 4. MAKE EXACTLY 5 SECONDS
    # --------------------------------------------------------

    target_length = int(
        CHUNK_DURATION * SAMPLE_RATE
    )

    if len(processed_audio) < target_length:

        processed_audio = np.pad(
            processed_audio,
            (
                0,
                target_length - len(processed_audio)
            )
        )

    elif len(processed_audio) > target_length:

        processed_audio = processed_audio[
            :target_length
        ]

    return processed_audio.astype(np.float32)


# ============================================================
# WHISPER WORKER
# ============================================================

def transcription_worker():

    global previous_audio

    print("🧠 Loading Faster-Whisper model...")

    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

    print("✓ Whisper model loaded.")
    print()

    overlap_samples = int(
        OVERLAP_DURATION * SAMPLE_RATE
    )

    while (
        not stop_event.is_set()
        or not audio_queue.empty()
    ):

        try:

            chunk_number, raw_audio = audio_queue.get(
                timeout=0.5
            )

        except queue.Empty:

            continue

        start_time = (
            chunk_number - 1
        ) * CHUNK_DURATION

        end_time = (
            chunk_number
        ) * CHUNK_DURATION

        temp_filename = None

        try:

            # =================================================
            # PREPROCESS CURRENT 5-SECOND CHUNK
            # =================================================

            print(
                f"⚙️ Preprocessing chunk {chunk_number}..."
            )

            processed_audio = preprocess_audio(
                raw_audio
            )

            print(
                f"✓ Chunk {chunk_number} preprocessing complete"
            )


            # =================================================
            # SAVE PROCESSED 5-SECOND CHUNK
            # =================================================

            processed_filename = os.path.join(
                PROCESSED_DIR,
                f"chunk_{chunk_number}_processed.wav"
            )

            sf.write(
                processed_filename,
                processed_audio,
                SAMPLE_RATE
            )

            print(
                f"💾 Saved processed audio: "
                f"{processed_filename}"
            )


            # =================================================
            # CREATE 1-SECOND OVERLAP
            # =================================================

            if previous_audio is not None:

                processing_audio = np.concatenate(
                    [
                        previous_audio,
                        processed_audio
                    ]
                )

                overlap_offset = OVERLAP_DURATION

                print(
                    f"🔄 Chunk {chunk_number}: "
                    f"1-second overlap added"
                )

            else:

                processing_audio = processed_audio

                overlap_offset = 0


            # =================================================
            # TEMPORARY WAV FOR WHISPER
            # =================================================

            temp_file = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            )

            temp_filename = temp_file.name

            temp_file.close()


            sf.write(
                temp_filename,
                processing_audio,
                SAMPLE_RATE
            )


            # =================================================
            # TRANSCRIPTION
            # =================================================

            segments, info = model.transcribe(

                temp_filename,

                language="en",

                condition_on_previous_text=False,

                temperature=0,

                vad_filter=True,

                beam_size=5
            )


            text_parts = []


            for segment in segments:

                # --------------------------------------------
                # Ignore segments completely inside overlap
                # --------------------------------------------

                if segment.end <= overlap_offset:

                    continue

                text = segment.text.strip()

                if text:

                    text_parts.append(text)


            transcript = " ".join(
                text_parts
            ).strip()


            # =================================================
            # SUSPICIOUS TERM DETECTION
            # =================================================

            detected_terms = []

            if transcript:

                detected_terms = detect_suspicious_terms(
                    transcript
                )


            # =================================================
            # DISPLAY CHUNK
            # =================================================

            print()
            print("=" * 60)

            print(
                f"📦 CHUNK {chunk_number} "
                f"[{start_time:02d}s - {end_time:02d}s]"
            )

            print("-" * 60)


            if not transcript:

                print("🔇 No speech detected.")


            else:

                print(
                    f"Transcript: {transcript}"
                )


                # --------------------------------------------
                # SAVE TO FULL CONVERSATION
                # --------------------------------------------

                conversation.append({

                    "chunk": chunk_number,

                    "start": start_time,

                    "end": end_time,

                    "transcript": transcript,

                    "suspicious_terms": detected_terms

                })


                # --------------------------------------------
                # SUSPICIOUS TERMS
                # --------------------------------------------

                if detected_terms:

                    print()
                    print(
                        "⚠ Suspicious terms detected:"
                    )

                    for term in detected_terms:

                        print(
                            f"   - {term}"
                        )

                        all_suspicious_terms.add(
                            term
                        )

                else:

                    print()
                    print(
                        "✓ No suspicious terms detected."
                    )


            print("=" * 60)


            # =================================================
            # SAVE LAST 1 SECOND FOR NEXT CHUNK
            # =================================================

            if len(processed_audio) >= overlap_samples:

                previous_audio = processed_audio[
                    -overlap_samples:
                ].copy()

            else:

                previous_audio = processed_audio.copy()


        except Exception as e:

            print()
            print(
                f"❌ Error processing chunk "
                f"{chunk_number}: {e}"
            )

        finally:

            # -----------------------------------------------
            # Delete temporary Whisper WAV
            # -----------------------------------------------

            if (
                temp_filename is not None
                and os.path.exists(temp_filename)
            ):

                os.remove(temp_filename)

            audio_queue.task_done()


# ============================================================
# START WHISPER WORKER
# ============================================================

worker_thread = threading.Thread(
    target=transcription_worker,
    daemon=True
)

worker_thread.start()


# ============================================================
# MICROPHONE
# ============================================================

chunk_number = 1


print("=" * 60)
print("VoxShield - Real-Time Voice Detection V4")
print("=" * 60)
print()

print("🎙️ Microphone is ready.")
print()
print(
    "Audio is captured continuously in 5-second chunks."
)
print(
    "Audio is preprocessed before Whisper transcription."
)
print(
    "Each processed chunk is saved to processed_chunks/."
)
print(
    "Whisper uses 1-second overlap between chunks."
)
print()
print("Press Ctrl+C to stop.")
print()


try:

    while True:

        start_time = (
            chunk_number - 1
        ) * CHUNK_DURATION

        end_time = (
            chunk_number
        ) * CHUNK_DURATION


        print(
            f"🎙️ Recording chunk {chunk_number} "
            f"[{start_time:02d}s - {end_time:02d}s]..."
        )


        # ====================================================
        # RECORD EXACTLY 5 SECONDS
        # ====================================================

        audio = sd.rec(

            int(
                CHUNK_DURATION *
                SAMPLE_RATE
            ),

            samplerate=SAMPLE_RATE,

            channels=CHANNELS,

            dtype="float32"
        )

        sd.wait()


        # ====================================================
        # CONVERT TO 1D
        # ====================================================

        audio = audio.flatten()


        # ====================================================
        # SEND RAW AUDIO TO PROCESSING QUEUE
        # ====================================================

        audio_queue.put(
            (
                chunk_number,
                audio.copy()
            )
        )


        print(
            f"✓ Chunk {chunk_number} captured "
            f"→ sent to processing queue"
        )


        chunk_number += 1


except KeyboardInterrupt:

    print()
    print("=" * 60)
    print("🛑 Stopping microphone...")
    print("=" * 60)

    stop_event.set()


# ============================================================
# WAIT FOR ALL CHUNKS
# ============================================================

print()
print("⏳ Processing remaining audio...")

audio_queue.join()


# ============================================================
# SORT CONVERSATION
# ============================================================

conversation.sort(
    key=lambda x: x["chunk"]
)


# ============================================================
# BUILD FULL TRANSCRIPT
# ============================================================

full_transcript_parts = []

for chunk in conversation:

    full_transcript_parts.append(
        chunk["transcript"]
    )


full_transcript = " ".join(
    full_transcript_parts
)


# ============================================================
# FINAL JSON RESULT
# ============================================================

result = {

    "full_transcript": full_transcript,

    "suspicious_terms": sorted(
        all_suspicious_terms
    ),

    "total_suspicious_terms": len(
        all_suspicious_terms
    ),

    "chunks": conversation

}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FULL CONVERSATION
# ============================================================

print()
print()
print("=" * 60)
print("📄 FULL CONVERSATION TRANSCRIPT")
print("=" * 60)
print()


if conversation:

    for chunk in conversation:

        print(
            f"[{chunk['start']:02d}s - "
            f"{chunk['end']:02d}s] "
            f"{chunk['transcript']}"
        )

else:

    print("No speech was detected.")


# ============================================================
# FINAL SECURITY ANALYSIS
# ============================================================

print()
print("=" * 60)
print("🚨 FINAL SECURITY ANALYSIS")
print("=" * 60)
print()


if all_suspicious_terms:

    print("Suspicious terms found:")

    for term in sorted(
        all_suspicious_terms
    ):

        print(
            f"   ⚠ {term}"
        )


    print()

    print(
        f"Total unique suspicious terms: "
        f"{len(all_suspicious_terms)}"
    )

else:

    print(
        "✓ No suspicious terms detected."
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)

print(
    f"✓ Result saved to: {OUTPUT_JSON}"
)

print(
    f"✓ Processed chunks saved in: {PROCESSED_DIR}/"
)

print("=" * 60)

print()
print(
    "✓ VoxShield conversation processing complete."
)