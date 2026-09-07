# VoxShield – Audio Processing

This module handles the **real-time audio processing and speech transcription pipeline** for VoxShield.

The pipeline captures microphone audio continuously, processes it in chunks, reduces silence and background noise, and sends the processed audio to **Faster-Whisper** for transcription. It also detects suspicious/scam-related terms and generates structured output for further integration with the VoxShield system.

## Features

* 🎤 Continuous microphone audio capture
* ⏱️ 5-second audio chunking
* 🔇 Silence removal
* 🔊 Background noise reduction
* 📈 Audio normalization
* 💾 Saving processed audio chunks
* 🔄 1-second overlap between consecutive chunks
* 🧠 Faster-Whisper speech transcription
* 📝 Timestamped conversation transcript
* 🚨 Suspicious/scam keyword detection
* 📦 JSON output for system integration
* ⚡ Background processing using a queue and worker thread

---

## Project Structure

```text
audio_processing/
│
├── 10_final_preprocess.py   # Audio preprocessing pipeline
├── realtime_v4.py            # Real-time processing + transcription
├── requirements.txt          # Required Python packages
└── README.md                 # Documentation
```

---

## Processing Pipeline

```text
Microphone
    │
    ▼
5-second Audio Chunk
    │
    ▼
Silence Removal
    │
    ▼
Noise Reduction
    │
    ▼
Normalization
    │
    ▼
Processed Audio Chunk
    │
    ├──────────────► Saved as WAV
    │
    ▼
1-second Overlap
    │
    ▼
Faster-Whisper
    │
    ▼
Timestamped Transcript
    │
    ▼
Suspicious Term Detection
    │
    ▼
JSON Output
```

---

## Audio Preprocessing

The preprocessing stage performs the following operations:

### 1. Resampling

Input audio is converted to:

* Sample rate: **16 kHz**
* Channels: **Mono**

This provides a consistent format for speech recognition.

### 2. Silence Removal

Silent portions of the audio are removed using:

```python
librosa.effects.trim()
```

with:

```python
top_db = 30
```

### 3. Noise Reduction

Background noise is reduced using the `noisereduce` library.

The current configuration uses:

```python
stationary=False
prop_decrease=0.7
```

### 4. Normalization

The processed audio is normalized using:

```python
librosa.util.normalize()
```

This helps maintain a consistent audio level before transcription.

### 5. Chunking

Audio is processed in **5-second chunks**.

A **1-second overlap** is maintained between consecutive chunks to reduce the possibility of losing words at chunk boundaries.

---

## Real-Time Processing

`realtime_v4.py` continuously records audio from the microphone.

Instead of waiting for transcription to finish before recording the next chunk, the system uses:

```text
Recording Thread
       │
       ▼
     Queue
       │
       ▼
Processing Worker
       │
       ▼
Faster-Whisper
```

This allows recording to continue while the previous audio chunk is being processed.

### Chunk Processing

For the first chunk:

```text
Processed Chunk 1 → Whisper
```

For subsequent chunks:

```text
Previous 1 second
        +
Current 5 seconds
        ↓
     Whisper
```

The overlap helps preserve words that may occur near the boundary between two chunks.

---

## Speech Transcription

The project uses **Faster-Whisper** for speech-to-text transcription.

The current model configuration is:

```python
MODEL_SIZE = "base"
```

The model runs using:

```text
Device: CPU
Compute type: int8
```

This configuration is suitable for running the system on a CPU-based laptop.

---

## Suspicious Term Detection

After transcription, the system checks the generated text for potentially suspicious or scam-related terms.

Examples include:

```text
OTP
PIN
CVV
password
bank account
account number
UPI
transaction
bank details
```

Detected terms are reported separately from the transcript.

The comparison is performed case-insensitively.

---

## Output

The system produces:

### Console Output

Example:

```text
[00s - 05s] Hello, we are calling you from the bank
[05s - 10s] Please provide your bank OTP and CVV number

Suspicious terms:
bank
OTP
CVV
```

### JSON Output

The system also generates structured JSON containing the processed conversation and detected suspicious terms.

This output can be consumed by other components of VoxShield for further scam detection and decision-making.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bondhuece25-ui/Voice-cloning-sih.git
```

### 2. Navigate to the audio processing module

```bash
cd Voice-cloning-sih/audio_processing
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Preprocessing Pipeline

Run:

```bash
python 10_final_preprocess.py
```

The script processes the configured input audio and generates the corresponding processed output.

---

## Running Real-Time Audio Processing

Run:

```bash
python realtime_v4.py
```

The program will:

1. Start microphone capture.
2. Record audio continuously.
3. Divide audio into 5-second chunks.
4. Remove silence.
5. Reduce background noise.
6. Normalize the audio.
7. Save processed chunks.
8. Transcribe the processed audio using Faster-Whisper.
9. Detect suspicious terms.
10. Generate structured JSON output.

---

## Requirements

The main Python dependencies are listed in `requirements.txt`.

Core libraries include:

```text
faster-whisper
sounddevice
soundfile
numpy
librosa
noisereduce
```

Python **3.10+** is recommended.

---

## Generated Files

The following files/folders may be generated while running the system:

```text
processed_chunks/
voxshield_result.json
```

These are runtime-generated files and should generally **not be committed to GitHub**.

Audio recordings such as `.wav` files should also not be committed unless specifically required for testing or dataset purposes.

---

## Role in VoxShield

The audio-processing module acts as the bridge between the **raw microphone input** and the downstream scam/voice analysis components.

```text
Raw Voice
   │
   ▼
Audio Processing
   │
   ├── Silence Removal
   ├── Noise Reduction
   ├── Normalization
   └── Chunking
   │
   ▼
Speech-to-Text
   │
   ▼
Transcript + Suspicious Terms
   │
   ▼
VoxShield Detection System
```

---

## Future Improvements

Potential improvements include:

* Real-time voice activity detection
* Improved noise suppression
* Speaker diarization
* Emotion/prosody analysis
* Voice-cloning/deepfake detection integration
* More advanced scam-language classification
* Streaming transcription
* GPU acceleration when available
* Integration with the main VoxShield backend

---

## Team Contribution

**Audio Signal Processing – Member 2**

Responsibilities include:

* Designing the real-time audio processing pipeline
* Implementing microphone capture
* Implementing 5-second audio chunking
* Implementing silence removal
* Implementing background noise reduction
* Implementing audio normalization
* Saving processed audio chunks
* Integrating Faster-Whisper transcription
* Implementing suspicious-term detection
* Generating structured JSON output
* Preparing processed audio for downstream VoxShield models

---

## License

This project is developed as part of an academic/project implementation for VoxShield.
